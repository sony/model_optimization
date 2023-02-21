# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from enum import Enum
from typing import Any, Dict, List

from model_compression_toolkit.core.common.target_platform import QuantizationMethod


class QuantizationTarget(Enum):
    Activation = "Activation"
    Weights = "Weights"


def mark_quantizer(quantization_target: QuantizationTarget = None,
                   quantization_method: List[QuantizationMethod] = None,
                   quantizer_type: Any = None):
    """
    A function to be used as decoration for all inferable quantizers (which inherit from BaseInferableQuantizer).
    By decorating a class with this decoration, we can define required static properties of the quantizer.

    Args:
        quantization_target: QuantizationTarget value which indicates what is the target for quantization to
            use the quantizer for.
        quantization_method: A list of QuantizationMethod values to indicate all type of quantization methods that the
            quantizer supports.
        quantizer_type: The type of the quantizer (quantization technique).
            This can differ, depending on the purpose the quantizer is for.

    Returns: A function that decorates a class object.

    """
    def mark(quantizer_class_object: BaseInferableQuantizer):
        """
        Initializes the parameters for the decorator.

        Args:
            quantizer_class_object: The class to be decorated.

        Returns: A decorated class.

        """
        quantizer_class_object.quantization_target = quantization_target
        quantizer_class_object.quantization_method = quantization_method
        quantizer_class_object.quantizer_type = quantizer_type

        return quantizer_class_object

    return mark


class BaseInferableQuantizer:

    def __init__(self):
        """
        This class is a base quantizer which defines an abstract
        function which any quantizer needs to implement.
        """
        pass

    def initialize_quantization(self,
                                tensor_shape: Any,
                                name: str,
                                layer: Any) -> Dict[Any, Any]:
        """
        Return a dictionary of quantizer parameters and their names.

        Args:
            tensor_shape: tensor shape of the quantized tensor.
            name: Tensor name.
            layer: Layer to quantize.

        Returns:
            Dictionary of parameters names to the variables.
        """
        return {}
