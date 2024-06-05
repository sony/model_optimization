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

import logging


class SimilarityCalculator:

    def __init__(self,
                 dataset_utils,
                 model_folding,
                 similarity_functions,
                 device: str=None):

        self.dataset_utils = dataset_utils
        self.model_folding = model_folding
        self.similarity_functions = similarity_functions
        self.device = device

    def compute_tensors_similarity(self,
                                   predictions: Tuple[Any, Any],
                                   similarity_metrics: Dict[str, Callable]) -> Dict[str, float]:
        """
        Compute metrics based on predictions.

        Args:
            predictions (Tuple[Any, Any]): A tuple of predictions from the floating-point and quantized models.
            similarity_metrics (Dict[str, Callable]): Custom metrics for output evaluation.

        Returns:
            Dict[str, float]: A dictionary of computed metrics.
        """
        f_pred, q_pred = predictions
        metrics = {k: v(f_pred, q_pred) for k, v in similarity_metrics.items()}
        return metrics

    def get_float_to_quantized_compare_points(self,
                                              quantized_model,
                                              float_model) -> Dict[str, str]:
        """
        Maps corresponding layers between the float and quantized models for comparison.

        Args:
            quantized_model: The quantized model.
            float_model: The float model.

        Returns:
            A dictionary mapping float model layer names to quantized model layer names.
        """
        quant_points_names = self.get_quant_compare_points(quantized_model)

        float_name2quant_name = {}

        float_layers_names = self.get_float_layers_names(float_model)

        for quant_point in quant_points_names:
            candidate_float_layer_name = self.get_float_candidate_layer(quant_compare_point=quant_point,
                                                                        quantized_model=quantized_model)
            if candidate_float_layer_name in float_layers_names:
                if candidate_float_layer_name not in float_name2quant_name:
                    float_name2quant_name[candidate_float_layer_name] = quant_point
                else:
                    logging.critical(f"Duplicate mapping found for layer: {candidate_float_layer_name}")
            else:
                logging.warning(f"Skipping point {quant_point}")

        return float_name2quant_name

    def compute_similarity_metrics(self,
                                   float_model,
                                   quantized_model,
                                   dataset: Callable,
                                   custom_similarity_metrics: Dict[str, Callable] = None,
                                   is_validation: bool = False) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:

        dataset = partial(self.dataset_utils.wrapped_dataset,
                          dataset=dataset,
                          is_validation=is_validation,
                          device=self.device)

        float_model = self.model_folding.create_float_folded_model(float_model=float_model,
                                                                   representative_dataset=dataset)

        metrics_to_compute = self.similarity_functions.get_default_metrics()
        if custom_similarity_metrics:
            assert isinstance(custom_similarity_metrics, dict), (
                f"custom_similarity_metrics should be a dictionary but is "
                f"{type(custom_similarity_metrics)}")
            metrics_to_compute.update(custom_similarity_metrics)

        float_name2quant_name = self.get_float_to_quantized_compare_points(float_model=float_model,
                                                                           quantized_model=quantized_model)

        output_metrics = {key: [] for key in metrics_to_compute.keys()}
        intermediate_metrics = {layer: {key: [] for key in metrics_to_compute.keys()} for layer in
                                float_name2quant_name.values()}

        for x in dataset():
            float_activations, quant_activations, float_predictions, quant_predictions = self.get_activations(
                float_model, quantized_model, float_name2quant_name, x)

            output_results = self.compute_tensors_similarity((float_predictions, quant_predictions), metrics_to_compute)
            for key in output_metrics:
                output_metrics[key].append(output_results[key])

            for float_layer, quant_layer in float_name2quant_name.items():
                intermediate_results = self.compute_tensors_similarity(
                    (float_activations[float_layer], quant_activations[quant_layer]),
                    metrics_to_compute)
                for key in intermediate_metrics[quant_layer]:
                    intermediate_metrics[quant_layer][key].append(intermediate_results[key])

        aggregated_output_metrics = {key: sum(value) / len(value) for key, value in output_metrics.items()}

        for layer_name, layer_metrics in intermediate_metrics.items():
            for similarity_name, similarity_values_list in layer_metrics.items():
                intermediate_metrics[layer_name][similarity_name] = sum(
                    intermediate_metrics[layer_name][similarity_name]) / len(
                    intermediate_metrics[layer_name][similarity_name])

        return aggregated_output_metrics, intermediate_metrics

    def get_activations(self, float_model, quantized_model, float_name2quant_name, data):
        raise NotImplementedError

    def get_quant_compare_points(self, quantized_model):
        raise NotImplementedError

    def get_float_candidate_layer(self,
                                  quant_compare_point,
                                  quantized_model):
        raise NotImplementedError

    def get_float_layers_names(self, float_model):
        raise NotImplementedError
