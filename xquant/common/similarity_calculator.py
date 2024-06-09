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
from functools import partial

from typing import Tuple, Any, Dict, Callable

from xquant.common.dataset_utils import DatasetUtils
from xquant.common.model_analyzer_utils import ModelAnalyzerUtils
from xquant.common.model_folding_utils import ModelFoldingUtils
from xquant.common.similarity_metrics import SimilarityFunctions
from xquant.logger import Logger


class SimilarityCalculator:
    """

    """

    def __init__(self,
                 dataset_utils: DatasetUtils,
                 model_folding: ModelFoldingUtils,
                 similarity_functions: SimilarityFunctions,
                 model_analyzer_utils: ModelAnalyzerUtils,
                 device: str = None):
        """

        Args:
            dataset_utils:
            model_folding:
            similarity_functions:
            model_analyzer_utils:
            device:
        """

        self.dataset_utils = dataset_utils
        self.model_folding = model_folding
        self.similarity_functions = similarity_functions
        self.model_analyzer_utils = model_analyzer_utils
        self.device = device

    @staticmethod
    def compute_tensors_similarity(predictions: Tuple[Any, Any],
                                   similarity_metrics: Dict[str, Callable]) -> Dict[str, float]:
        """
        Compute the similarity between the compared tensors.

        Args:
            predictions: Tensors to compare.
            similarity_metrics: A dictionary with similarity metrics names and functions that compute the similarity.

        Returns:
            A dictionary of the similarity functions names and their values.
        """
        f_pred, q_pred = predictions
        metrics = {k: v(f_pred, q_pred) for k, v in similarity_metrics.items()}
        return metrics

    def _get_float_to_quantized_compare_points(self,
                                              quantized_model: Any,
                                              float_model: Any) -> Dict[str, str]:
        """
        Maps corresponding layers between the float and quantized models for comparison.

        Args:
            quantized_model: The quantized model.
            float_model: The float model.

        Returns:
            A dictionary mapping float model layer names to quantized model layer names.
        """
        quant_points_names = self.model_analyzer_utils.get_quant_compare_points(quantized_model)

        float_name2quant_name = {}

        float_layers_names = self.model_analyzer_utils.get_float_layers_names(float_model)

        for quant_point in quant_points_names:
            candidate_float_layer_name = self.model_analyzer_utils.get_float_candidate_layer(quant_compare_point=quant_point,
                                                                        quantized_model=quantized_model)
            if candidate_float_layer_name in float_layers_names:
                if candidate_float_layer_name not in float_name2quant_name:
                    float_name2quant_name[candidate_float_layer_name] = quant_point
                else:
                    Logger.get_logger().critical(f"Duplicate mapping found for layer: {candidate_float_layer_name}")
            else:
                Logger.get_logger().warning(f"Skipping point {quant_point}")

        return float_name2quant_name

    def compute_similarity_metrics(self,
                                   float_model: Any,
                                   quantized_model: Any,
                                   dataset: Callable,
                                   custom_similarity_metrics: Dict[str, Callable] = None,
                                   is_validation: bool = False) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:

        dataset = partial(self.dataset_utils.prepare_dataset,
                          dataset=dataset,
                          is_validation=is_validation,
                          device=self.device)

        float_model = self.model_folding.create_float_folded_model(float_model=float_model,
                                                                   representative_dataset=dataset)

        similarity_metrics_to_compute = self.similarity_functions.get_default_similarity_metrics()
        if custom_similarity_metrics:
            if not isinstance(custom_similarity_metrics, dict):
                Logger.get_logger().critical(
                    f"custom_similarity_metrics should be a dictionary but is of type "
                    f"{type(custom_similarity_metrics)}.")
            similarity_metrics_to_compute.update(custom_similarity_metrics)

        float_name2quant_name = self._get_float_to_quantized_compare_points(float_model=float_model,
                                                                            quantized_model=quantized_model)

        output_similarity_metrics = {key: [] for key in similarity_metrics_to_compute.keys()}
        intermediate_similarity_metrics = {layer: {key: [] for key in similarity_metrics_to_compute.keys()} for layer in
                                           float_name2quant_name.values()}

        for x in dataset():
            float_activations, quant_activations, float_predictions, quant_predictions = self.model_analyzer_utils.get_activations(
                float_model, quantized_model, float_name2quant_name, x)

            output_results = self.compute_tensors_similarity((float_predictions, quant_predictions), similarity_metrics_to_compute)
            for key in output_similarity_metrics:
                output_similarity_metrics[key].append(output_results[key])

            for float_layer, quant_layer in float_name2quant_name.items():
                intermediate_results = self.compute_tensors_similarity(
                    (float_activations[float_layer], quant_activations[quant_layer]),
                    similarity_metrics_to_compute)
                for key in intermediate_similarity_metrics[quant_layer]:
                    intermediate_similarity_metrics[quant_layer][key].append(intermediate_results[key])

        aggregated_output_similarity_metrics = {key: sum(value) / len(value) for key, value in output_similarity_metrics.items()}

        for layer_name, layer_similarity_metrics in intermediate_similarity_metrics.items():
            for similarity_name, similarity_values_list in layer_similarity_metrics.items():
                intermediate_similarity_metrics[layer_name][similarity_name] = sum(
                    intermediate_similarity_metrics[layer_name][similarity_name]) / len(
                    intermediate_similarity_metrics[layer_name][similarity_name])

        return aggregated_output_similarity_metrics, intermediate_similarity_metrics

