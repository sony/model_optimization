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
from typing import Any, Dict, List, Tuple

from abc import ABC, abstractmethod

from model_compression_toolkit.logger import Logger


class ModelAnalyzer(ABC):
    """
    This class provides abstract methods for analyzing a model, specifically for
    extracting activations and comparing float and quantized models.
    """

    @abstractmethod
    def extract_model_activations(self,
                                  float_model: Any,
                                  quantized_model: Any,
                                  float_name2quant_name: Dict[str, str],
                                  data: List[Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extracts activations from both the float and quantized models.

        Args:
            float_model: The float model.
            quantized_model: The quantized model.
            float_name2quant_name: A mapping from float model layer names to quantized model layer
            names.
            data: Input data for which to compute activations.

        Returns:
                - Dictionary of activations for the float model.
                - Dictionary of activations for the quantized model.
        """
        Logger.critical("This method should be implemented by the framework-specific ModelAnalyzer.")  # pragma: no cover


    @abstractmethod
    def identify_quantized_compare_points(self, quantized_model: Any) -> List[str]:
        """
        Identifies the layers in the quantized model that are wrapped with the quantization wrapper.
        These layers will serve as comparison points.

        Notes:
            This currently means that the quantized compare points are the linear layers that are wrapped,
            but this may be changed in the future.

        Args:
            quantized_model: The quantized model from which to identify comparison points.

        Returns:
            List[str]: Names of the layers wrapped with the quantization wrapper.
        """
        Logger.critical("This method should be implemented by the framework-specific ModelAnalyzer.")  # pragma: no cover


    @abstractmethod
    def find_corresponding_float_layer(self,
                                       quant_compare_point: List[str],
                                       quantized_model: Any) -> str:
        """
        Finds the corresponding float model layer for a given quantized model layer.

        Args:
            quant_compare_point: The name of the quantized model layer.
            quantized_model: The quantized model.

        Returns:
            str: The name of the corresponding layer in the float model.
        """
        Logger.critical("This method should be implemented by the framework-specific ModelAnalyzer.")  # pragma: no cover

    @abstractmethod
    def extract_float_layer_names(self, float_model: Any) -> List[str]:
        """
        Extracts the names of all layers in the float model.

        Args:
            float_model: The float model from which to extract layer names.

        Returns:
            List[str]: Names of all layers in the float model.
        """
        Logger.critical("This method should be implemented by the framework-specific ModelAnalyzer.")  # pragma: no cover


