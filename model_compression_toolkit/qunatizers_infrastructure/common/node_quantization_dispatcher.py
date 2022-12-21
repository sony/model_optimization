# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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

from typing import Dict, List

from model_compression_toolkit.qunatizers_infrastructure.common.base_quantizer import BaseQuantizer


class NodeQuantizationDispatcher:
    def __init__(self,
                 weight_quantizers: Dict[str, BaseQuantizer] = None,
                 activation_quantizers: List[BaseQuantizer] = None):
        """
        Node quantization dispatcher collects all the quantizer of a given layer.

        Args:
            weight_quantizers: A dictionary between weight name to it quantizer .
            activation_quantizers: A list of activation quantization one for each layer output.
        """
        self.weight_quantizers = weight_quantizers if weight_quantizers is not None else dict()
        self.activation_quantizers = activation_quantizers if activation_quantizers is not None else list()

    def add_weight_quantizer(self, param_name: str, quantizer: BaseQuantizer):
        """
        This function add a weight quantizer to existing node dispatcher

        Args:
            param_name: The name of the parameter to quantize
            quantizer: A quantizer.

        Returns: None

        """
        self.weight_quantizers.update({param_name: quantizer})

    @property
    def is_activation_quantization(self) -> bool:
        """
        This function check activation quantizer exists in dispatcher.
        Returns: a boolean if activation quantizer exists

        """
        return len(self.activation_quantizers) > 0

    @property
    def is_weights_quantization(self) -> bool:
        """
        This function check weights quantizer exists in dispatcher.

        Returns: a boolean if weights quantizer exists

        """
        return len(self.weight_quantizers) > 0
