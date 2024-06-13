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
from typing import List, Tuple, Dict

from mct_quantizers import KerasQuantizationWrapper
from model_compression_toolkit.xquant.common.model_analyzer import ModelAnalyzer
import keras
import numpy as np


class KerasModelAnalyzer(ModelAnalyzer):
    """
    This class provides utilities for analyzing Keras models, specifically for
    extracting activations and comparing float and quantized models.
    """

    def extract_model_activations(self,
                                  float_model: keras.Model,
                                  quantized_model: keras.Model,
                                  float_name2quant_name: Dict[str, str],
                                  data: List[np.ndarray]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Extracts activations from both the float and quantized models.

        Args:
            float_model (keras.Model): The float model.
            quantized_model (keras.Model): The quantized model.
            float_name2quant_name (Dict[str, str]): A mapping from float model layer names to quantized model layer
            names.
            data (List[np.ndarray]): Input data for which to compute activations.

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
                - Dictionary of activations for the float model.
                - Dictionary of activations for the quantized model.
                - Predictions from the float model.
                - Predictions from the quantized model.
        """

        def _compute_activations(model: keras.Model, layer_names: List[str], data: List[np.ndarray]) -> Tuple[
            Dict[str, np.ndarray], np.ndarray]:
            """
            Computes the activations for the specified layers of the model, and the model's output.

            Args:
                model (keras.Model): The model from which to extract activations.
                layer_names (List[str]): Names of the layers for which to compute activations.
                data (List[np.ndarray]): Input data for the model.

            Returns:
                Tuple[Dict[str, np.ndarray], np.ndarray]:
                    - Dictionary mapping layer names to their corresponding activations.
                    - Predictions from the model.
            """
            # Extract the outputs of the specified layers plus the model output
            _model_outputs = [model.get_layer(name).output for name in layer_names] + [model.output]

            # Create a new model that outputs the intermediate and final layer outputs
            intermediate_layer_model = keras.Model(inputs=model.input, outputs=_model_outputs)
            predictions = intermediate_layer_model(data)

            # Map layer names to their corresponding activations and return with the output predictions
            return {layer_name: predictions[i].numpy() for i, layer_name in enumerate(layer_names)}, predictions[-1].numpy()

        # Compute activations for the quantized model
        quant_activations, quant_predictions = _compute_activations(quantized_model,
                                                                    list(float_name2quant_name.values()),
                                                                    data)
        # Compute activations for the float model
        float_activations, float_predictions = _compute_activations(float_model,
                                                                    list(float_name2quant_name.keys()),
                                                                    data)

        # Concatenate predictions if they are lists.
        if isinstance(quant_predictions, list):
            quant_predictions = np.concatenate(quant_predictions)
        if isinstance(float_predictions, list):
            float_predictions = np.concatenate(float_predictions)

        # Return activations and predictions for both models
        return float_activations, quant_activations, float_predictions, quant_predictions

    def identify_quantized_compare_points(self, quantized_model: keras.Model) -> List[str]:
        """
        Identifies the layers in the quantized model that are wrapped with the quantization wrapper.
        These layers will serve as comparison points.

        Notes:
            This currently means that the quantized compare points are the linear layers that are wrapped,
            but this may be changed in the future.

        Args:
            quantized_model (keras.Model): The quantized model from which to identify comparison points.

        Returns:
            List[str]: Names of the layers wrapped with the quantization wrapper.
        """
        return [layer.name for layer in quantized_model.layers if isinstance(layer, KerasQuantizationWrapper)]

    def find_corresponding_float_layer(self,
                                       quant_compare_point: str,
                                       quantized_model: keras.Model) -> str:
        """
        Finds the corresponding float model layer for a given quantized model layer.

        Args:
            quant_compare_point (str): The name of the quantized model layer.
            quantized_model (keras.Model): The quantized model.

        Returns:
            str: The name of the corresponding layer in the float model.
        """
        return quantized_model.get_layer(quant_compare_point).layer.name

    def extract_float_layer_names(self, float_model: keras.Model) -> List[str]:
        """
        Extracts the names of all layers in the float model.

        Args:
            float_model (keras.Model): The float model from which to extract layer names.

        Returns:
            List[str]: Names of all layers in the float model.
        """
        float_layers_names = [layer.name for layer in float_model.layers]
        return float_layers_names
