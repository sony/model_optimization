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
from typing import List

from model_compression_toolkit.core import FrameworkInfo
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.quantization.quantization_fn_selection import get_activation_quantization_fn
from model_compression_toolkit.core.keras.back2framework.keras_model_builder import KerasModelBuilder
from tensorflow.python.util.object_identity import Reference as TFReference


class QuantizedKerasModelBuilder(KerasModelBuilder):
    """
    Builder of quantized Keras models.
    """

    def __init__(self,
                 graph: common.Graph,
                 append2output=None,
                 return_float_outputs: bool = False):
        """

        Args:
            graph: Graph to build the model from.
            append2output: Nodes to append to model's output.
            return_float_outputs: Whether the model returns float tensors or not.
        """

        super().__init__(graph,
                         append2output,
                         return_float_outputs)

    def _quantize_node_activations(self,
                                   node: BaseNode,
                                   input_tensors: List[TFReference]) -> List[TFReference]:
        """
        Quantize node's activation given input tensors.

        Args:
            node: Node to quantize its outputs.
            input_tensors: Input tensors of the node.

        Returns:
            Output of the node.

        """
        activation_quantizer = get_activation_quantization_fn(node.final_activation_quantization_cfg)
        return activation_quantizer(input_tensors)
