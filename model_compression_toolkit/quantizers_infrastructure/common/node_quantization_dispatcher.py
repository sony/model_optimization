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

from model_compression_toolkit.quantizers_infrastructure import BaseInferableQuantizer
from model_compression_toolkit.core.common.logger import Logger

class NodeQuantizationDispatcher:
    def __init__(self,
                 weight_quantizers: Dict[str, BaseInferableQuantizer] = None,
                 activation_quantizers: List[BaseInferableQuantizer] = None):
        """
        Node quantization dispatcher collects all the quantizer of a given layer.

        Args:
            weight_quantizers: A dictionary between weight name to it quantizer .
            activation_quantizers: A list of activation quantization one for each layer output.
        """
        self.weight_quantizers = weight_quantizers if weight_quantizers is not None else dict()
        self.activation_quantizers = activation_quantizers if activation_quantizers is not None else list()


    def set_weight_quantizers(self, weight_quantizers: Dict[str, BaseInferableQuantizer]):
        """
        This function sets weight quantizers to existing node dispatcher

        Args:
            weight_quantizers: weight quantizers

        Returns: None
        """
        for name,quantizer in weight_quantizers.items():
            if not isinstance(quantizer, BaseInferableQuantizer):
                Logger.error(f"quantizer is supposed to be BaseInferableQuantizer but it's not!")  # pragma: no cover
        self.weight_quantizers = weight_quantizers

    def set_activation_quantizers(self, activation_quantizers: List[BaseInferableQuantizer]):
        """
        This function sets activation quantizers to existing node dispatcher

        Args:
            activation_quantizers: activation quantizers

        Returns: None
        """
        for quantizer in activation_quantizers:
            if not isinstance(quantizer, BaseInferableQuantizer):
                Logger.error(f"quantizer is supposed to be BaseInferableQuantizer but it's not!")  # pragma: no cover
        self.activation_quantizers = activation_quantizers


    def add_weight_quantizer(self, param_name: str, quantizer: BaseInferableQuantizer):
        """
        This function adds a weight quantizer to existing node dispatcher

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
        return self.num_act_quantizers > 0

    @property
    def is_weights_quantization(self) -> bool:
        """
        This function check weights quantizer exists in dispatcher.

        Returns: a boolean if weights quantizer exists

        """
        return self.num_weight_quantizers > 0

    @property
    def num_weight_quantizers(self):
        return len(self.weight_quantizers)


    @property
    def num_act_quantizers(self):
        return len(self.activation_quantizers)

