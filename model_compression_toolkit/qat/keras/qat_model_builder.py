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
from tensorflow.keras.layers import Layer, InputLayer
from tensorflow.python.util.object_identity import Reference as TFReference

from model_compression_toolkit import get_target_platform_capabilities
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.constants import TENSORFLOW
from model_compression_toolkit.core.common.framework_info import FrameworkInfo

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

    if node.is_weights_quantization_enabled() and not fw_info.is_kernel_op(node.type):
        common.Logger.error("QAT Error: Quantizing a node without a kernel isn't supported")
    return node.is_weights_quantization_enabled() or node.is_activation_quantization_enabled()


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
