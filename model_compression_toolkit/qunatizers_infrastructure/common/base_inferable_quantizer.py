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
from typing import Any, Dict


class QuantizationTarget(Enum):
    Activation = 0
    Weights = 1


class BaseInferableQuantizer:
    def __init__(self,
                 quantization_target: QuantizationTarget):
        """
        This class is a base quantizer which defines an abstract
        function which any quantizer needs to implement.

        Args:
            quantization_target: A enum which selects the quantizer tensor type: activation or weights.
        """
        self.quantization_target = quantization_target

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
