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
import torch
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import BaseNode
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
import model_compression_toolkit.qunatizers_infrastructure.pytorch.quantize_wrapper as qi
from model_compression_toolkit.qat.pytorch.quantizer.quantization_dispatcher_builder import quantization_dispatcher_builder
from model_compression_toolkit.core.common.framework_info import FrameworkInfo


def _is_qat_applicable(node: common.BaseNode,
                       fw_info: FrameworkInfo) -> bool:
    """
    A function for deciding if a layer should be fine-tuned during QAT
    Args:
        node (BaseNode): Node for quantization decision
        fw_info (FrameworkInfo): Pytorch quantization information

    Returns:
        A boolean whether the layer is to be wrapped with a QuantizeWrapper
    """

    return (fw_info.is_kernel_op(node.type) and node.is_weights_quantization_enabled()) or node.is_activation_quantization_enabled()


def qat_wrapper(n: common.BaseNode, module: torch.nn.Module):
    """
    A function which takes a computational graph node and a pytorch module and perform the quantization wrapping
    Args:
        n: A node of mct graph.
        module: A Pytorch module

    Returns: Wrapped layer

    """
    if _is_qat_applicable(n, DEFAULT_PYTORCH_INFO):
        return qi.PytorchQuantizationWrapper(module, quantization_dispatcher_builder(n, DEFAULT_PYTORCH_INFO))
    else:
        return module
