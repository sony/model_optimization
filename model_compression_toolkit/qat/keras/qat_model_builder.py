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
from tensorflow.keras.layers import Layer
from tensorflow.python.util.object_identity import Reference as TFReference

from model_compression_toolkit import get_target_platform_capabilities
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.constants import TENSORFLOW
from model_compression_toolkit.core.common.framework_info import FrameworkInfo

from model_compression_toolkit.core.keras.back2framework.keras_model_builder import KerasModelBuilder
from model_compression_toolkit.core.keras.constants import DEFAULT_TP_MODEL
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.qat.keras.quantizer.quantization_dispatcher_builder import \
    quantization_dispatcher_builder
from model_compression_toolkit import qunatizers_infrastructure as qi

DEFAULT_KERAS_TPC = get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)


def _is_qat_applicable(node: common.BaseNode,
                       fw_info: FrameworkInfo) -> bool:
    """
    A function for deciding if a layer should be fine-tuned during QAT
    Args:
        node (BaseNode): Node for quantization decision
        fw_info (FrameworkInfo): Keras quantization information

    Returns:
        A boolean whether the layer is to be wrapped with a QuantizeWrapper
    """

    return fw_info.is_kernel_op(node.type) and node.is_weights_quantization_enabled()


def qat_wrapper(n: common.BaseNode, layer: Layer):
    """
    A function which takes a computational graph node and a keras layer and perform the quantization wrapping
    Args:
        n: A node of mct graph.
        layer: A keras layer

    Returns: Wrapped layer

    """
    if _is_qat_applicable(n, DEFAULT_KERAS_INFO):
        return qi.KerasQuantizationWrapper(layer, quantization_dispatcher_builder(n, DEFAULT_KERAS_INFO))
    else:
        return layer


class QATKerasModelBuilder(KerasModelBuilder):
    """
    Builder of QAT Keras models.
    """

    def __init__(self,
                 graph: common.Graph,
                 append2output=None,
                 fw_info: FrameworkInfo = DEFAULT_KERAS_INFO,
                 return_float_outputs: bool = False):
        """

        Args:
            graph: Graph to build the model from.
            append2output: Nodes to append to model's output.
            fw_info: Information about the specific framework of the model that is built.
            return_float_outputs: Whether the model returns float tensors or not.
        """
        super().__init__(graph,
                         append2output,
                         fw_info,
                         return_float_outputs,
                         wrapper=qat_wrapper)

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
        return node.final_activation_quantization_cfg.quantize_node_output(input_tensors)
