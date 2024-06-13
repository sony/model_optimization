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

from model_compression_toolkit.xquant.common.constants import MODEL_OUTPUT_KEY
from model_compression_toolkit.xquant.common.dataset_utils import DatasetUtils
from model_compression_toolkit.xquant.common.model_analyzer import ModelAnalyzer
from model_compression_toolkit.xquant.common.model_folding_utils import ModelFoldingUtils
from model_compression_toolkit.xquant.common.similarity_functions import SimilarityFunctions
from model_compression_toolkit.logger import Logger

class SimilarityCalculator:
    """
    A class to calculate the similarity between two models (that are often referred as float
    and quantized models). It utilizes various utility classes for dataset preparation, model folding,
    similarity computation, and model analysis.
    """

    def __init__(self,
                 dataset_utils: DatasetUtils,
                 model_folding: ModelFoldingUtils,
                 similarity_functions: SimilarityFunctions,
                 model_analyzer_utils: ModelAnalyzer,
                 device: str = None):
        """
        Initialize the SimilarityCalculator with required utilities.

        Args:
            dataset_utils (DatasetUtils): Utility class for dataset preparation.
            model_folding (ModelFoldingUtils): Utility class for model folding operations.
            similarity_functions (SimilarityFunctions): Class containing similarity functions.
            model_analyzer_utils (ModelAnalyzer): Utility class for model analysis.
            device (str, optional): Device to perform computations on (e.g., 'cpu', 'cuda'). Defaults to None.
        """
        self.dataset_utils = dataset_utils
        self.model_folding = model_folding
        self.similarity_functions = similarity_functions
        self.model_analyzer_utils = model_analyzer_utils
        self.device = device

    @staticmethod
    def compute_tensors_similarity(tensors_to_compare: Tuple[Any, Any],
                                   similarity_metrics: Dict[str, Callable]) -> Dict[str, float]:
        """
        Compute the similarity between two tensors using provided similarity metrics.

        Args:
            tensors_to_compare (Tuple[Any, Any]): Tensors to compare by computing their similarity.
            similarity_metrics (Dict[str, Callable]): A dictionary with similarity metric names and functions.

        Returns:
            Dict[str, float]: A dictionary of similarity metric names and their computed values.
        """
        x, y = tensors_to_compare
        similarity_metrics = {k: v(x, y) for k, v in similarity_metrics.items()}
        return similarity_metrics

    def _get_float_to_quantized_compare_points(self,
                                               quantized_model: Any,
                                               float_model: Any) -> Dict[str, str]:
        """
        Map corresponding layers between the float and quantized models for comparison.

        Args:
            quantized_model (Any): The quantized model.
            float_model (Any): The float model.

        Returns:
            Dict[str, str]: A dictionary mapping float model layer names to quantized model layer names.
        """
        # Identify the points in the quantized model to compare.
        quant_points_names = self.model_analyzer_utils.identify_quantized_compare_points(quantized_model)

        float_name2quant_name = {}

        # Extract the names of the layers in the float model.
        float_layers_names = self.model_analyzer_utils.extract_float_layer_names(float_model)

        # Map each quantized layer to the corresponding float layer.
        for quant_point in quant_points_names:
            candidate_float_layer_name = self.model_analyzer_utils.find_corresponding_float_layer(
                quant_compare_point=quant_point, quantized_model=quantized_model)

            if candidate_float_layer_name in float_layers_names:
                if candidate_float_layer_name not in float_name2quant_name:
                    float_name2quant_name[candidate_float_layer_name] = quant_point
                else:
                    Logger.critical(f"Duplicate mapping found for layer: {candidate_float_layer_name}.")
            else:
                Logger.warning(
                    f"Could not find a matching layer in the float model for layer with name {quant_point}, "
                    f"skipping it in similarity metrics comparison points computation.")

        return float_name2quant_name

    def compute_similarity_metrics(self,
                                   float_model: Any,
                                   quantized_model: Any,
                                   dataset: Callable,
                                   custom_similarity_metrics: Dict[str, Callable] = None,
                                   is_validation: bool = False) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Compute the similarity metrics between the two models (usually, float and quantized models).

        Args:
            float_model (Any): The float model.
            quantized_model (Any): The quantized model.
            dataset (Callable): A callable to provide the dataset.
            custom_similarity_metrics (Dict[str, Callable], optional): Custom similarity metrics. Defaults to None.
            is_validation (bool, optional): Flag to indicate if the dataset is for validation. Defaults to False.

        Returns:
            Tuple[Dict[str, float], Dict[str, Dict[str, float]]]: Aggregated output similarity metrics and
            intermediate similarity metrics for each layer.
        """
        # Prepare the dataset such that the rest of operations are indistinguishable between the representative
        # dataset and the validation dataset.
        dataset = partial(self.dataset_utils.prepare_dataset,
                          dataset=dataset,
                          is_validation=is_validation,
                          device=self.device)

        # Create a folded version of the float model.
        float_model = self.model_folding.create_float_folded_model(float_model=float_model,
                                                                   representative_dataset=dataset)

        # Gather similarity metrics to compute (default and custom).
        similarity_metrics_to_compute = self.similarity_functions.get_default_similarity_metrics()
        if custom_similarity_metrics:
            if not isinstance(custom_similarity_metrics, dict):
                Logger.critical(
                    f"custom_similarity_metrics should be a dictionary but is of type "
                    f"{type(custom_similarity_metrics)}.")
            similarity_metrics_to_compute.update(custom_similarity_metrics)

        # Map float model layers to quantized model layers for comparison.
        float_name2quant_name = self._get_float_to_quantized_compare_points(float_model=float_model,
                                                                            quantized_model=quantized_model)

        # Initialize dictionaries to store similarity metrics.
        output_similarity_metrics = {key: [] for key in similarity_metrics_to_compute.keys()}
        intermediate_similarity_metrics = {layer: {key: [] for key in similarity_metrics_to_compute.keys()} for layer in
                                           float_name2quant_name.values()}

        # Iterate over the dataset and compute similarity metrics.
        for x in dataset():
            # Extract activations and predictions from both models.
            float_activations, quant_activations = (
                self.model_analyzer_utils.extract_model_activations(
                float_model, quantized_model, float_name2quant_name, x))

            float_predictions = float_activations[MODEL_OUTPUT_KEY]
            quant_predictions = quant_activations[MODEL_OUTPUT_KEY]

            # Compute similarity metrics for the output predictions.
            output_results = self.compute_tensors_similarity((float_predictions, quant_predictions),
                                                             similarity_metrics_to_compute)
            for key in output_similarity_metrics:
                output_similarity_metrics[key].append(output_results[key])

            # Compute similarity metrics for each intermediate layer.
            for float_layer, quant_layer in float_name2quant_name.items():
                intermediate_results = self.compute_tensors_similarity(
                    (float_activations[float_layer], quant_activations[quant_layer]),
                    similarity_metrics_to_compute)
                for key in intermediate_similarity_metrics[quant_layer]:
                    intermediate_similarity_metrics[quant_layer][key].append(intermediate_results[key])

        # Aggregate the output similarity metrics.
        aggregated_output_similarity_metrics = {key: sum(value) / len(value) for key, value in
                                                output_similarity_metrics.items()}

        # Aggregate the intermediate similarity metrics for each layer.
        for layer_name, layer_similarity_metrics in intermediate_similarity_metrics.items():
            for similarity_name, similarity_values_list in layer_similarity_metrics.items():
                if len(similarity_values_list) == 0:
                    Logger.critical(f"Can not average similarities of an empty list.")
                intermediate_similarity_metrics[layer_name][similarity_name] = sum(similarity_values_list) / len(similarity_values_list)

        return aggregated_output_similarity_metrics, intermediate_similarity_metrics
