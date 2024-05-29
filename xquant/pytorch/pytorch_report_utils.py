#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

from keras import Model
from model_compression_toolkit.core.common.quantization.quantization_config import DEFAULTCONFIG
from tqdm import tqdm
from typing import Any, Callable, Dict

from mct_quantizers import PytorchQuantizationWrapper
from model_compression_toolkit.core.common.model_collector import ModelCollector
from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter
from model_compression_toolkit.core.pytorch.reader.reader import model_reader
from model_compression_toolkit.ptq.pytorch.quantization_facade import DEFAULT_PYTORCH_TPC
from xquant import XQuantConfig
from xquant.common.constants import XQUANT_VAL, XQUANT_REPR, INTERMEDIATE_METRICS_REPR, INTERMEDIATE_METRICS_VAL
from xquant.common.framework_report_utils import FrameworkReportUtils, MSE_METRIC_NAME, CS_METRIC_NAME, SQNR_METRIC_NAME
from functools import partial
import torch
import numpy as np
import tensorflow as tf

from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation

from xquant.logger import Logger


class PytorchReportUtils(FrameworkReportUtils):
    def __init__(self, report_dir: str):
        """
        Args:
            report_dir: Logging dir path.
        """
        tb_writer = TensorboardWriter(report_dir, DEFAULT_PYTORCH_INFO)
        super().__init__(tb_writer=tb_writer)

    def get_metric_on_output(self,
                             float_model: Model,
                             quantized_model: Model,
                             dataset: Callable,
                             custom_metrics_output: Dict[str, Callable] = None,
                             is_validation: bool = False):
        """
        Computes metrics on the outputs of the float and quantized models.

        Args:
            float_model: The original float model.
            quantized_model: The quantized model.
            dataset: A function that returns the dataset.
            custom_metrics_output: A dictionary of custom metrics to compute.
            is_validation: A flag indicating if this is a validation dataset.

        Returns:
            A dictionary with aggregated metrics.
        """

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dataset = partial(self.wrapped_dataset, dataset=dataset, is_validation=is_validation, device=device)

        float_model.to(device)
        quantized_model.to(device)
        float_model.eval()
        quantized_model.eval()

        # Get the default metrics and add any custom metrics
        metrics_to_compute = list(self.get_default_metrics().keys())
        if custom_metrics_output:
            assert isinstance(custom_metrics_output,
                              dict), (f"custom_metrics_output should be a dictionary but is "
                                      f"{type(custom_metrics_output)}")
            metrics_to_compute += list(custom_metrics_output.keys())

        # Initialize a dictionary to store metrics
        metrics = {key: [] for key in metrics_to_compute}

        # Iterate over the dataset and compute predictions
        for x in dataset():
            with torch.no_grad():
                float_pred = float_model(*x)
                quant_pred = quantized_model(*x)
                predictions = (float_pred, quant_pred)

            # Compute and store metrics
            results = self.compute_metrics(predictions, custom_metrics_output)
            for key in metrics:
                metrics[key].append(results[key])

        # Aggregate metrics by averaging
        aggregated_metrics = {key: sum(value) / len(value) for key, value in metrics.items()}

        return aggregated_metrics

    def get_metric_on_intermediate(self,
                                   float_model: Model,
                                   quantized_model: Model,
                                   dataset: Callable,
                                   custom_metrics_intermediate: Dict[str, Callable] = None,
                                   is_validation: bool = False):
        """
        Computes metrics on the intermediate layers of the float and quantized models.

        Args:
            float_model: The original float model.
            quantized_model: The quantized model.
            dataset: A function that returns the dataset.
            custom_metrics_intermediate: A dictionary of custom metrics to compute.
            is_validation: A flag indicating if this is a validation dataset.

        Returns:
            A dictionary with aggregated metrics for each layer.
        """

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dataset = partial(self.wrapped_dataset, dataset=dataset, is_validation=is_validation, device=device)

        float_model = self.create_float_folded_model(float_model=float_model,
                                                     representative_dataset=dataset)

        float_model.to(device)
        quantized_model.to(device)
        float_model.eval()
        quantized_model.eval()

        # Get the default metrics and add any custom metrics
        metrics_to_compute = list(self.get_default_metrics().keys())
        if custom_metrics_intermediate:
            if not isinstance(custom_metrics_intermediate, dict):
                Logger.critical(
                    f"custom_metrics_output should be a dictionary but is {type(custom_metrics_intermediate)}")
            metrics_to_compute += list(custom_metrics_intermediate.keys())

        # Map layers between the float and quantized models
        float_name2quant_name = self.get_float_to_quantized_compare_points(float_model=float_model,
                                                                           quantized_model=quantized_model)

        def get_activation(name: str, activations: dict):
            """
            Returns a hook function to capture activations.

            Args:
                name: The name of the layer.
                activations: Dictionary to store activations.

            Returns:
                Hook function to register.
            """

            def hook(model, input, output):
                activations[name] = output.detach()

            return hook

        activations_float = {}
        activations_quant = {}

        # Register hooks for the float model
        for layer_name in float_name2quant_name.keys():
            layer = dict([*float_model.named_modules()])[layer_name]
            layer.register_forward_hook(get_activation(layer_name, activations_float))

        # Register hooks for the quantized model
        for layer_name in float_name2quant_name.values():
            layer = dict([*quantized_model.named_modules()])[layer_name]
            layer.register_forward_hook(get_activation(layer_name, activations_quant))

        results = {}
        for float_layer, quant_layer in float_name2quant_name.items():
            results[quant_layer] = []

        # Iterate over the dataset and compute activations
        for x in dataset():
            with torch.no_grad():
                _ = (float_model(*x), quantized_model(*x))

            # Compute and store metrics for each layer
            for float_layer, quant_layer in float_name2quant_name.items():
                results[quant_layer].append(
                    self.compute_metrics((activations_float[float_layer], activations_quant[quant_layer]),
                                         custom_metrics_intermediate))

        # Aggregate metrics by averaging
        aggregated_metrics = {}
        for layer_name, layer_metrics in results.items():
            combined_dict = {}
            for item in layer_metrics:
                for key, value in item.items():
                    if key not in combined_dict:
                        combined_dict[key] = []
                    combined_dict[key].append(value)
            for k, v in combined_dict.items():
                combined_dict[k] = np.mean(v)
            aggregated_metrics[layer_name] = combined_dict

        return aggregated_metrics

    def get_default_metrics(self) -> Dict[str, Callable]:
        """
        Returns default metrics to be computed on model outputs.

        Returns:
            A dictionary of metric names and their corresponding functions.
        """

        def compute_mse(f_pred: tf.Tensor, q_pred: tf.Tensor) -> float:
            """
            Computes Mean Squared Error between float and quantized predictions.

            Args:
                f_pred: Float model predictions.
                q_pred: Quantized model predictions.

            Returns:
                Mean Squared Error as a float.
            """
            mse = torch.nn.functional.mse_loss(f_pred, q_pred)
            return mse.item()

        def compute_cs(f_pred: tf.Tensor, q_pred: tf.Tensor) -> float:
            """
            Computes Cosine Similarity between float and quantized predictions.

            Args:
                f_pred: Float model predictions.
                q_pred: Quantized model predictions.

            Returns:
                Cosine Similarity as a float.
            """
            cs = torch.nn.functional.cosine_similarity(f_pred.flatten(), q_pred.flatten(), dim=0)
            return cs.item()

        def compute_sqnr(f_pred: tf.Tensor, q_pred: tf.Tensor) -> float:
            """
            Computes Signal-to-Quantization-Noise Ratio between float and quantized predictions.

            Args:
                f_pred: Float model predictions.
                q_pred: Quantized model predictions.

            Returns:
                Signal-to-Quantization-Noise Ratio as a float.
            """
            signal_power = torch.mean(f_pred ** 2)
            noise_power = torch.mean((f_pred - q_pred) ** 2)
            sqnr = signal_power / noise_power
            return sqnr.item()

        return {MSE_METRIC_NAME: compute_mse,
                CS_METRIC_NAME: compute_cs,
                SQNR_METRIC_NAME: compute_sqnr}

    def create_float_folded_model(self,
                                  float_model: Model,
                                  representative_dataset: Callable,
                                  ) -> Model:
        """
        Creates a folded float model based on the representative dataset and core configuration.

        Args:
            float_model: The original float model.
            representative_dataset: A function that returns the representative dataset.

        Returns:
            The folded float model.
        """
        float_graph = self.convert_to_graph(float_model,
                                            representative_dataset)
        float_folded_model, _ = PytorchImplementation().model_builder(float_graph,
                                                                      mode=ModelBuilderMode.FLOAT,
                                                                      append2output=None,
                                                                      fw_info=DEFAULT_PYTORCH_INFO)
        return float_folded_model

    def convert_to_graph(self,
                         model: torch.nn.Module,
                         repr_dataset: Callable):
        """
        Convert Pytorch model to networkx graph representation.

        Args:
            model: Keras model to convert.
            repr_dataset: Representative dataset (not used in Keras).

        Returns:
            Graph representing the model.
        """
        graph = graph_preparation_runner(in_model=model,
                                         representative_data_gen=repr_dataset,
                                         fw_impl=PytorchImplementation(),
                                         fw_info=DEFAULT_PYTORCH_INFO,
                                         quantization_config=DEFAULTCONFIG,
                                         tpc=DEFAULT_PYTORCH_TPC)
        return graph

    def wrapped_dataset(self,
                        dataset: Any,
                        is_validation: bool,
                        device: str):
        """
        Wraps the dataset to ensure it is properly transferred to the device and processed on a given device.

        Args:
            dataset: The dataset to be wrapped.
            is_validation: A flag indicating if this is a validation dataset.
            device: The device to transfer the data to.

        Returns:
            A generator that yields processed data.
        """

        def process_data(data: Any, is_validation: bool, device: str):
            """
            Processes individual data samples to transfer them to the device.

            Args:
                data: The data sample to process.
                is_validation: A flag indicating if this is a validation dataset.
                device: The device to transfer the data to.

            Returns:
                A generator that yields the processed data.
            """

            def transfer_to_device(_data):
                if isinstance(_data, np.ndarray):
                    return torch.from_numpy(_data).to(device)
                return _data.to(device)

            if is_validation:
                inputs = data[0]  # Assume data[0] contains the inputs and data[1] the labels
                if isinstance(inputs, list):
                    data = [transfer_to_device(t) for t in inputs]
                else:
                    data = [transfer_to_device(inputs)]
            else:
                data = [transfer_to_device(t) for t in data]

            yield data

        for x in dataset():
            return process_data(x, is_validation, device)

    def get_float_to_quantized_compare_points(self,
                                              quantized_model: Model,
                                              float_model: Model) -> Dict[str, str]:
        """
        Maps corresponding layers between the float and quantized models for comparison.

        Args:
            quantized_model: The quantized model.
            float_model: The float model.

        Returns:
            A dictionary mapping float model layer names to quantized model layer names.
        """
        quant_points_names = [n for n, m in quantized_model.named_modules() if
                              isinstance(m, PytorchQuantizationWrapper)]
        float_name2quant_name = {}

        for quant_point in quant_points_names:
            candidate_float_layer_name = quant_point
            if candidate_float_layer_name in [n for n, m in float_model.named_modules()]:
                if candidate_float_layer_name not in float_name2quant_name:
                    float_name2quant_name[candidate_float_layer_name] = quant_point
                else:
                    Logger.critical(f"Duplicate mapping found for layer: {candidate_float_layer_name}")
            else:
                Logger.warning(f"Skipping point {quant_point}")

        return float_name2quant_name

    def get_quantized_graph(self,
                            quantized_model: torch.nn.Module,
                            repr_dataset: Callable):
        """
        Get a graph representation of the quantized model.

        Args:
            quantized_model: The quantized model.
            repr_dataset: Representative dataset to use during the graph building.

        Returns:
            Graph representation of the quantized model.
        """

        pytorch_impl = PytorchImplementation()
        graph = model_reader(quantized_model,
                             representative_data_gen=repr_dataset,
                             to_tensor=pytorch_impl.to_tensor,
                             to_numpy=pytorch_impl.to_numpy)
        return graph

    def get_quant_graph_with_metrics(self,
                                     quantized_model: torch.nn.Module,
                                     collected_data: Dict[str, Any],
                                     xquant_config: XQuantConfig,
                                     repr_dataset: Callable):
        """
        Updates the quantized model graph with metrics data collected during evaluation.

        Args:
            quantized_model: The quantized model.
            collected_data: Dictionary containing the collected metrics data.
            xquant_config: Configuration settings for quantization.
            repr_dataset: Representative dataset used during graph building.

        Returns:
            The updated quantized model graph.
        """
        quant_graph = self.get_quantized_graph(quantized_model=quantized_model,
                                               repr_dataset=repr_dataset)
        for node in quant_graph.nodes:
            if xquant_config.compute_intermediate_metrics_repr:
                if node.name in collected_data[INTERMEDIATE_METRICS_REPR].keys():
                    node.framework_attr[XQUANT_REPR] = collected_data[INTERMEDIATE_METRICS_REPR][f"{node.name}"]
                elif node.name.removesuffix("_layer") in collected_data[INTERMEDIATE_METRICS_REPR].keys():
                    node.framework_attr[XQUANT_REPR] = collected_data[INTERMEDIATE_METRICS_REPR][
                        node.name.removesuffix("_layer")]

            if xquant_config.compute_intermediate_metrics_val:
                if node.name in collected_data[INTERMEDIATE_METRICS_VAL].keys():
                    node.framework_attr[XQUANT_VAL] = collected_data[INTERMEDIATE_METRICS_VAL][f"{node.name}"]
                elif node.name.removesuffix("_layer") in collected_data[INTERMEDIATE_METRICS_VAL].keys():
                    node.framework_attr[XQUANT_VAL] = collected_data[INTERMEDIATE_METRICS_VAL][
                        node.name.removesuffix("_layer")]
        return quant_graph

    def add_histograms_to_tensorboard(self,
                                      model: torch.nn.Module,
                                      repr_dataset: Callable,):
        """
        Collect histograms and add them to Tensorboard.

        Args:
            model: Model to collect histograms on.
            repr_dataset: Dataset that is used an input for collecting histograms.

        Returns:
            None
        """

        graph = self.convert_to_graph(model=model,
                                      repr_dataset=repr_dataset)
        mi = ModelCollector(graph,
                            PytorchImplementation(),
                            DEFAULT_PYTORCH_INFO)
        for _data in tqdm(repr_dataset(), "Statistics Collection"):
            mi.infer(_data)

        self.tb_writer.add_histograms(graph, "")