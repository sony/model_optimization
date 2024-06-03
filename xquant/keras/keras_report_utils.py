#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
#
import keras
from model_compression_toolkit.core.common.quantization.quantization_config import DEFAULTCONFIG
from tensorflow.keras.models import Model
from tqdm import tqdm

from mct_quantizers import KerasQuantizationWrapper
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.common.model_collector import ModelCollector
from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter
from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation

from typing import Any, Dict, Callable, List
import numpy as np
from keras import Model
import tensorflow as tf
from functools import partial

from xquant import XQuantConfig
from xquant.common.constants import INTERMEDIATE_METRICS_REPR, INTERMEDIATE_METRICS_VAL, XQUANT_REPR, XQUANT_VAL
from xquant.common.framework_report_utils import FrameworkReportUtils, MSE_METRIC_NAME, CS_METRIC_NAME, SQNR_METRIC_NAME
from model_compression_toolkit.ptq.keras.quantization_facade import DEFAULT_KERAS_TPC
from model_compression_toolkit.core.keras.reader.reader import model_reader

from xquant.keras.similarity_metrics import KerasSimilarityMetrics
from xquant.logger import Logger


class KerasReportUtils(FrameworkReportUtils):

    def __init__(self, report_dir: str):
        """
        Args:
            report_dir: Logging dir path.
        """
        tb_writer = TensorboardWriter(report_dir, DEFAULT_KERAS_INFO)
        self.similarity_metrics = KerasSimilarityMetrics()
        super().__init__(tb_writer=tb_writer)

    def get_metric_on_output(self,
                             float_model: keras.Model,
                             quantized_model: keras.Model,
                             dataset: Callable,
                             custom_metrics_output: Dict[str, Callable] = None,
                             is_validation: bool = False) -> Dict[str, float]:
        """
        Compute metrics on the output of the model.

        Args:
            float_model (keras.Model): The floating-point Keras model.
            quantized_model (keras.Model): The quantized Keras model.
            dataset (Callable): Dataset used for inference.
            custom_metrics_output (Dict[str, Callable], optional): Custom metrics for output evaluation. Defaults to
            None.
            is_validation (bool, optional): Flag indicating if this is a validation dataset. Defaults to False.

        Returns:
            Dict[str, float]: A dictionary of computed metrics.
        """

        dataset = partial(self.wrapped_dataset,
                          dataset=dataset,
                          is_validation=is_validation)

        metrics_to_compute = self.similarity_metrics.get_default_metrics()
        if custom_metrics_output:
            if not isinstance(custom_metrics_output, dict):
                Logger.critical(f"custom_metrics_output should be a dictionary but is {type(custom_metrics_output)}")
            metrics_to_compute.update(custom_metrics_output)

        metrics = {key: [] for key in list(metrics_to_compute.keys())}

        for x in dataset():
            float_predictions = float_model.predict(x)
            quant_predictions = quantized_model.predict(x)
            results = self.compute_metrics((float_predictions, quant_predictions),
                                           metrics_to_compute)

            # Accumulating results
            for key in metrics:
                metrics[key].append(results[key])

        # Averaging metrics across the dataset
        aggregated_metrics = {key: sum(value) / len(value) for key, value in metrics.items()}

        return aggregated_metrics

    def get_metric_on_intermediate(self,
                                   float_model: keras.Model,
                                   quantized_model: keras.Model,
                                   dataset: Callable,
                                   custom_metrics_intermediate: Dict[str, Callable] = None,
                                   is_validation: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics on intermediate layers of the model.

        Args:
            float_model (keras.Model): The floating-point Keras model.
            quantized_model (keras.Model): The quantized Keras model.
            dataset (Callable): Dataset used for inference.
            custom_metrics_intermediate (Dict[str, Callable], optional): Custom metrics for intermediate layers.
            Defaults to None.
            is_validation (bool, optional): Flag indicating if this is a validation dataset. Defaults to False.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary of computed metrics for intermediate layers.
        """

        float_model = self.create_float_folded_model(float_model=float_model,
                                                     representative_dataset=None)

        dataset = partial(self.wrapped_dataset,
                          dataset=dataset,
                          is_validation=is_validation)

        def get_activations(model: Model,
                            layer_names: List[str],
                            data: Any) -> Dict[str, np.ndarray]:
            """
            Extract activations from specified layers of the model for the given input data.

            Args:
                model (Model): The Keras model from which to extract activations.
                layer_names (List[str]): List of layer names for which activations are to be extracted.
                data (Any): Input data for which activations are to be computed.

            Returns:
                Dict[str, np.ndarray]: A dictionary mapping layer names to their corresponding activations.
            """
            # Create a new model that outputs the activations of the specified layers
            intermediate_layer_model = Model(inputs=model.input,
                                             outputs=[model.get_layer(name).output for name in layer_names])

            predictions = intermediate_layer_model.predict(data)

            # Map the activations to their corresponding layer names
            return {layer_name: predictions[i] for i, layer_name in enumerate(layer_names)}



        metrics_to_compute = self.similarity_metrics.get_default_metrics()
        if custom_metrics_intermediate:
            if not isinstance(custom_metrics_intermediate, dict):
                Logger.critical(f"custom_metrics_intermediate should be a dictionary but is {type(custom_metrics_intermediate)}")
            metrics_to_compute.update(custom_metrics_intermediate)

        float_name2quant_name = self.get_float_to_quantized_compare_points(float_model=float_model,
                                                                           quantized_model=quantized_model)

        results = {q_layer: [] for q_layer in float_name2quant_name.values()}

        for x in dataset():
            quant_activations = get_activations(quantized_model, list(float_name2quant_name.values()), x)
            float_activations = get_activations(float_model, list(float_name2quant_name.keys()), x)

            for float_layer, quant_layer in float_name2quant_name.items():
                float_activation = float_activations[float_layer]
                quant_activation = quant_activations[quant_layer]

                results[quant_layer].append(
                    self.compute_metrics((float_activation, quant_activation), metrics_to_compute))

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

    def create_float_folded_model(self,
                                  float_model: keras.Model,
                                  representative_dataset: Any,
                                  ) -> keras.Model:
        """
        Create a folded version of the floating-point model.

        Args:
            float_model (keras.Model): The original floating-point model.
            representative_dataset (Any): Representative dataset used during quantization.

        Returns:
            keras.Model: The folded floating-point model.
        """
        float_graph = self.convert_to_graph(model=float_model)
        float_folded_model, _ = KerasImplementation().model_builder(float_graph,
                                                                    mode=ModelBuilderMode.FLOAT,
                                                                    append2output=None,
                                                                    fw_info=DEFAULT_KERAS_INFO)
        return float_folded_model

    def convert_to_graph(self,
                         model: Model,
                         repr_dataset: Callable=None):
        """
        Convert Keras model to networkx graph representation.

        Args:
            model: Keras model to convert.
            repr_dataset: Representative dataset (not used in Keras).

        Returns:
            Graph representing the model.
        """
        graph = graph_preparation_runner(in_model=model,
                                         representative_data_gen=repr_dataset,
                                         quantization_config=DEFAULTCONFIG,
                                         fw_info=DEFAULT_KERAS_INFO,
                                         fw_impl=KerasImplementation(),
                                         tpc=DEFAULT_KERAS_TPC)
        return graph

    def wrapped_dataset(self,
                        dataset: Any,
                        is_validation: bool,
                        device: str = None) -> Any:
        """
        Generator function that wraps 'dataset' to be able to handle both
        representative and validation datasets.

        Args:
            dataset (Any): The dataset to wrap.
            is_validation (bool): Flag indicating if this is a validation dataset.
            device (str, optional): Device to use for inference. Defaults to None.

        Yields:
            Any: Processed data from the dataset.
        """

        def process_data(x: Any, is_validation: bool) -> Any:
            return x[0] if is_validation else x

        for x in dataset():
            yield process_data(x, is_validation)

    def get_float_to_quantized_compare_points(self,
                                              quantized_model: keras.Model,
                                              float_model: keras.Model) -> Dict[str, str]:
        """
        Get comparison points between the floating-point and quantized models.

        Args:
            quantized_model (keras.Model): The quantized model.
            float_model (keras.Model): The floating-point model.

        Returns:
            Dict[str, str]: A dictionary mapping comparison points between the two models.
        """
        quant_points_names = [
            layer.name for layer in quantized_model.layers
            if isinstance(layer, KerasQuantizationWrapper)
        ]

        float_name2quant_name = {}

        for quant_point in quant_points_names:
            candidate_float_layer_name = quantized_model.get_layer(quant_point).layer.name

            if candidate_float_layer_name in [layer.name for layer in float_model.layers]:
                if candidate_float_layer_name not in float_name2quant_name:
                    float_name2quant_name[candidate_float_layer_name] = quant_point
                else:
                    Logger.critical(f"Duplicate mapping found for layer: {candidate_float_layer_name}")
            else:
                Logger.warning(f"Skipping point {quant_point}")

        return float_name2quant_name

    def get_quantized_graph(self,
                            quantized_model: Model,
                            repr_dataset: Callable = None):
        """
        Get a graph representation of the quantized model.

        Args:
            quantized_model: The quantized model.
            repr_dataset: Representative dataset to use during the graph building (not used in Keras).

        Returns:
            Graph representation of the quantized model.
        """
        quant_graph = model_reader(quantized_model)
        return quant_graph


    def get_quant_graph_with_metrics(self,
                                     quantized_model: keras.Model,
                                     collected_data: Dict[str, Any],
                                     xquant_config: XQuantConfig,
                                     repr_dataset: Callable = None) -> Graph:
        """
        Generate the quantized graph with associated metrics.

        Args:
            quantized_model (keras.Model): The quantized Keras model.
            collected_data (Dict[str, Any]): Data collected during quantization.
            xquant_config (XQuantConfig): Configuration settings for explainable quantization.
            repr_dataset (Callable): Representative dataset (not used in Keras).

        Returns:
            Graph: A graph structure with metrics.
        """
        quant_graph = self.get_quantized_graph(quantized_model=quantized_model)
        for node in quant_graph.nodes:
            if node.name in collected_data[INTERMEDIATE_METRICS_REPR].keys():
                node.framework_attr[XQUANT_REPR] = collected_data[INTERMEDIATE_METRICS_REPR][node.name]
            if node.name in collected_data[INTERMEDIATE_METRICS_VAL].keys():
                node.framework_attr[XQUANT_VAL] = collected_data[INTERMEDIATE_METRICS_VAL][node.name]
        return quant_graph

    def add_histograms_to_tensorboard(self,
                                      model: Model,
                                      repr_dataset: Callable):
        """
        Collect histograms and add them to Tensorboard.

        Args:
            model: Model to collect histograms on.
            repr_dataset: Dataset that is used an input for collecting histograms.

        Returns:
            None
        """
        graph = self.convert_to_graph(model,
                                      repr_dataset)
        mi = ModelCollector(graph,
                            KerasImplementation(),
                            DEFAULT_KERAS_INFO)
        for _data in tqdm(repr_dataset(), "Collecting Histograms"):
            mi.infer(_data)

        self.tb_writer.add_histograms(graph, "")

