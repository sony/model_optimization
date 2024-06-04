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


from typing import Any, Callable, Dict

from mct_quantizers import PytorchQuantizationWrapper
from model_compression_toolkit.core.pytorch.reader.reader import model_reader
from model_compression_toolkit.ptq.pytorch.quantization_facade import DEFAULT_PYTORCH_TPC
from xquant import XQuantConfig
from xquant.common.constants import XQUANT_VAL, XQUANT_REPR, INTERMEDIATE_METRICS_REPR, INTERMEDIATE_METRICS_VAL
from xquant.common.framework_report_utils import FrameworkReportUtils
from functools import partial
import torch
import numpy as np
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from xquant.common.model_folding_utils import ModelFoldingUtils
from xquant.common.tensorboard_utils import TensorboardUtils

from xquant.pytorch.dataset_utils import PytorchDatasetUtils
from xquant.pytorch.similarity_metrics import PytorchSimilarityMetrics


class PytorchReportUtils(FrameworkReportUtils):
    def __init__(self, report_dir: str):
        """
        Args:
            report_dir: Logging dir path.
        """
        fw_info = DEFAULT_PYTORCH_INFO
        fw_impl = PytorchImplementation()

        similarity_metrics = PytorchSimilarityMetrics()
        dataset_utils = PytorchDatasetUtils()
        model_folding = ModelFoldingUtils(fw_info=fw_info,
                                          fw_impl=fw_impl,
                                          fw_default_tpc=DEFAULT_PYTORCH_TPC)
        tb_utils = TensorboardUtils(report_dir=report_dir,
                                    fw_impl=fw_impl,
                                    fw_info=fw_info,
                                    model_folding_utils=model_folding)
        super().__init__(fw_info=fw_info,
                         fw_impl=fw_impl,
                         tb_utils=tb_utils,
                         dataset_utils=dataset_utils,
                         similarity_metrics=similarity_metrics,
                         model_folding=model_folding)

    def get_metric_on_output(self,
                             float_model: torch.nn.Module,
                             quantized_model: torch.nn.Module,
                             dataset: Callable,
                             custom_similarity_metrics: Dict[str, Callable] = None,
                             is_validation: bool = False):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dataset = partial(self.dataset_utils.wrapped_dataset, dataset=dataset, is_validation=is_validation,
                          device=device)

        float_model.to(device)
        quantized_model.to(device)
        float_model.eval()
        quantized_model.eval()

        # Get the default metrics and add any custom metrics
        metrics_to_compute = self.similarity_metrics.get_default_metrics()
        if custom_similarity_metrics:
            assert isinstance(custom_similarity_metrics,
                              dict), (f"custom_similarity_metrics should be a dictionary but is "
                                      f"{type(custom_similarity_metrics)}")
            metrics_to_compute.update(custom_similarity_metrics)

        # Initialize a dictionary to store metrics
        metrics = {key: [] for key in metrics_to_compute.keys()}

        # Iterate over the dataset and compute predictions
        for x in dataset():
            with torch.no_grad():
                float_pred = float_model(*x)
                quant_pred = quantized_model(*x)
                predictions = (float_pred, quant_pred)

            # Compute and store metrics
            results = self.compute_metrics(predictions, metrics_to_compute)
            for key in metrics:
                metrics[key].append(results[key])

        # Aggregate metrics by averaging
        aggregated_metrics = {key: sum(value) / len(value) for key, value in metrics.items()}

        return aggregated_metrics

    def get_metric_on_intermediate(self,
                                   float_model: torch.nn.Module,
                                   quantized_model: torch.nn.Module,
                                   dataset: Callable,
                                   custom_similarity_metrics: Dict[str, Callable] = None,
                                   is_validation: bool = False):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dataset = partial(self.dataset_utils.wrapped_dataset, dataset=dataset, is_validation=is_validation,
                          device=device)

        float_model = self.model_folding.create_float_folded_model(float_model=float_model,
                                                                   representative_dataset=dataset)

        float_model.to(device)
        quantized_model.to(device)
        float_model.eval()
        quantized_model.eval()

        # Get the default metrics and add any custom metrics
        metrics_to_compute = self.similarity_metrics.get_default_metrics()
        if custom_similarity_metrics:
            if not isinstance(custom_similarity_metrics, dict):
                Logger.critical(
                    f"custom_similarity_metrics should be a dictionary but is {type(custom_similarity_metrics)}")
            metrics_to_compute.update(custom_similarity_metrics)

        # Map layers between the float and quantized models
        float_name2quant_name = self.get_float_to_quantized_compare_points(float_model=float_model,
                                                                           quantized_model=quantized_model)

        def get_activation(name: str, activations: dict):

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
                float_pred = float_model(*x)
                quant_pred = quantized_model(*x)

            # Compute and store metrics for each layer
            for float_layer, quant_layer in float_name2quant_name.items():
                results[quant_layer].append(
                    self.compute_metrics((activations_float[float_layer], activations_quant[quant_layer]),
                                         metrics_to_compute))

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

    def get_float_to_quantized_compare_points(self,
                                              quantized_model: torch.nn.Module,
                                              float_model: torch.nn.Module) -> Dict[str, str]:
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
        quant_graph = model_reader(quantized_model,
                                   representative_data_gen=repr_dataset,
                                   to_tensor=self.fw_impl.to_tensor,
                                   to_numpy=self.fw_impl.to_numpy)

        for node in quant_graph.nodes:
            if node.name in collected_data[INTERMEDIATE_METRICS_REPR].keys():
                node.framework_attr[XQUANT_REPR] = collected_data[INTERMEDIATE_METRICS_REPR][f"{node.name}"]
            elif node.name.removesuffix("_layer") in collected_data[INTERMEDIATE_METRICS_REPR].keys():
                node.framework_attr[XQUANT_REPR] = collected_data[INTERMEDIATE_METRICS_REPR][
                    node.name.removesuffix("_layer")]

            if node.name in collected_data[INTERMEDIATE_METRICS_VAL].keys():
                node.framework_attr[XQUANT_VAL] = collected_data[INTERMEDIATE_METRICS_VAL][f"{node.name}"]
            elif node.name.removesuffix("_layer") in collected_data[INTERMEDIATE_METRICS_VAL].keys():
                node.framework_attr[XQUANT_VAL] = collected_data[INTERMEDIATE_METRICS_VAL][
                    node.name.removesuffix("_layer")]
        return quant_graph
