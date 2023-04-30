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
from typing import Tuple
import numpy as np
import torch
from torch.nn import Conv2d
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common.substitutions.residual_collapsing import ResidualCollapsing
from model_compression_toolkit.core.pytorch.constants import KERNEL
from model_compression_toolkit.logger import Logger


def residual_collapsing_node_matchers() -> Tuple[NodeOperationMatcher, NodeOperationMatcher]:
    """
    Function generates matchers for matching:
    (Conv2D, Add)[activation=linear] -> Conv2D.
    Returns:
        Matcher for Conv2D node and Add node
    """
    first_node = NodeOperationMatcher(Conv2d)
    second_node = NodeOperationMatcher(torch.add)
    return first_node, second_node


def residual_collapsing_fn(first_node: BaseNode,
                           kernel_str: str) -> np.ndarray:
    """
    Collapsing residual addition into convolution
    Inspired by https://arxiv.org/pdf/2103.09404.pdf - Algorithm2
    Args:
        first_node: First layer node to collapse into.
        kernel_str: The framework specific attribute name of the convolution layer's weight/kernel.
    Returns:
        The modified layer node's weights: kernel
    """
    if first_node.type == Conv2d:
        # Get nodes attributes
        kernel = first_node.get_weights_by_keys(kernel_str)
        (Cout, Cin, kH, kW) = kernel.shape

        # Collapsing residual by adding "1" to kernel diagonal
        idxH = (kH - 1) // 2
        idxW = (kW - 1) // 2
        for i in range(Cout):
            kernel[i, i, idxH, idxW] += 1
        return kernel
    else:
        Logger.error("No supported add residual collapsing for {}".format(first_node.type))


def pytorch_residual_collapsing() -> ResidualCollapsing:
    """
    Returns:
        A ResidualCollapsing initialized for pytorch models.
    """
    first_node, second_node = residual_collapsing_node_matchers()
    return ResidualCollapsing(first_node,
                              second_node,
                              residual_collapsing_fn,
                              KERNEL)
