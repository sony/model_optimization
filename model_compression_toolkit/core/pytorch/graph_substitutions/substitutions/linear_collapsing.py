# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
import torch.nn.functional as F
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, torch_tensor_to_numpy
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.substitutions.linear_collapsing import Conv2DCollapsing
from model_compression_toolkit.core.pytorch.constants import KERNEL, KERNEL_SIZE, STRIDES, DILATIONS, BIAS, USE_BIAS, FILTERS, PADDING, GROUPS
from model_compression_toolkit.core.common.logger import Logger


def linear_collapsing_node_matchers() -> Tuple[NodeOperationMatcher, NodeOperationMatcher]:
    """
    Function generates matchers for matching:
    (Conv2D, Conv2D)[activation=linear] -> Conv2D.
    Returns:
        Matcher for 2 consecutive linear convolution
    """
    first_node = NodeOperationMatcher(Conv2d)
    second_node = NodeOperationMatcher(Conv2d)
    return first_node, second_node


def conv2d_collapsing_fn(first_node: BaseNode,
                         second_node: BaseNode,
                         kernel_str: str,
                         bias_str: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collapsing 2 convolutions to one convolution: Out = k2*(k1*x+b1)+b2 = k2*k1*x+k2*b1+b2 = k*x+b
    We calculate k=k2*k1 (collapsed kernel) by injecting identity tensor to the convolutions and extract the output
    We calculate b=k2*b1+b2 (collapsed bias) matrix multiplication
    Args:
        first_node: First layer node to collapse to second layer node
        second_node: Second layer node
        kernel_str: The framework specific attribute name of the convolution layer's weight/kernel.
        bias_str: The framework specific attribute name of the convolution layer's bias.
    Returns:
        The modified layer node's weights: kernel, bias
    """
    if first_node.type == Conv2d and second_node.type == Conv2d:
        # Get nodes attributes
        kernel1 = first_node.get_weights_by_keys(kernel_str)
        kernel2 = second_node.get_weights_by_keys(kernel_str)
        bias1 = first_node.get_weights_by_keys(bias_str)
        bias2 = second_node.get_weights_by_keys(bias_str)
        strides1 = first_node.framework_attr[STRIDES]
        strides2 = second_node.framework_attr[STRIDES]

        # --------------------------------------- #
        # Kernel collapsing: k=k2*k1
        # --------------------------------------- #
        # Inspired by https://arxiv.org/pdf/2103.09404.pdf - Algorithm1

        # Generate identity input with padding
        kx, ky = kernel1.shape[2] + kernel2.shape[2] - 1, kernel1.shape[3] + kernel2.shape[3] - 1
        x_pad, y_pad = 2 * kx - 1, 2 * ky - 1
        in_tensor = torch.eye(kernel1.shape[1])
        in_tensor = torch.unsqueeze(torch.unsqueeze(in_tensor, 2), 3)
        in_tensor = F.pad(in_tensor, (int(np.ceil((x_pad - 1) / 2)),
                                      int(np.floor((x_pad - 1) / 2)),
                                      int(np.ceil((y_pad - 1) / 2)),
                                      int(np.floor((y_pad - 1) / 2))))

        # Run first Conv2D
        conv1_out = F.conv2d(input=to_torch_tensor(in_tensor), weight=to_torch_tensor(kernel1), stride=strides1, padding=(0,0))

        # Run second Conv2D
        kernel2_torch = to_torch_tensor(kernel2)
        conv2_out = F.conv2d(input=conv1_out, weight=kernel2_torch, stride=strides2)

        # Extract collapsed kernel from output: the collapsed kernel is the output of the convolution after fixing the dimension
        kernel_collapsed = torch_tensor_to_numpy(torch.permute(torch.flip(conv2_out,[3,2]), dims=[1,0,2,3]))

        # --------------------------------------- #
        # Bias collapsing: b=k2*b1+b2
        # --------------------------------------- #
        bias_collapsed = None
        if bias1 is not None:
            bias1_torch = to_torch_tensor(bias1)
            bias_collapsed = torch_tensor_to_numpy(torch.matmul(torch.sum(kernel2_torch,dim=(2, 3)), bias1_torch))
            if bias2 is not None:
                bias_collapsed += bias2
        elif bias2 is not None:
            bias_collapsed = bias2

        return kernel_collapsed, bias_collapsed
    else:
        Logger.error("No supported layer collapsing of {} and {}".format(first_node.type, second_node.type))


def pytorch_linear_collapsing() -> Conv2DCollapsing:
    """
    Returns:
        A Conv2DCollapsing initialized for pytorch models.
    """
    first_node, second_node = linear_collapsing_node_matchers()
    return Conv2DCollapsing(first_node,
                            second_node,
                            conv2d_collapsing_fn,
                            KERNEL,
                            KERNEL_SIZE,
                            BIAS,
                            USE_BIAS,
                            STRIDES,
                            PADDING,
                            DILATIONS,
                            GROUPS,
                            FILTERS)
