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
import numpy as np
from torch.nn import BatchNorm2d, Conv2d, ConvTranspose2d

from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.substitutions.batchnorm_folding import BatchNormalizationFolding, BatchNormalizationForwardFolding
from model_compression_toolkit.core.pytorch.constants import KERNEL, BIAS, GAMMA, BETA, MOVING_MEAN, MOVING_VARIANCE, \
    EPSILON, USE_BIAS, GROUPS, IN_CHANNELS, OUT_CHANNELS


def batchnorm_folding_node_matchers() -> [BaseNode, BaseNode]:
    """
    Function generates matchers for matching:
    (Conv2d, ConvTranspose2d)-> BatchNorm2d.

    Returns:
        Matcher for batch norm nodes, and source nodes.
    """
    bn_node = NodeOperationMatcher(BatchNorm2d)
    source_node = NodeOperationMatcher(Conv2d) | \
                  NodeOperationMatcher(ConvTranspose2d)
    return bn_node, source_node


def batchnorm_forward_folding_node_matchers() -> [BaseNode, BaseNode]:
    """
    Function generates matchers for matching:
    (BatchNormalization, dw-Conv2d 1x1) --> (Conv2d, ConvTranspose2d)

    Returns:
        Matcher for batch norm nodes, and source nodes.
    """
    bn_or_dw1x1_node = NodeOperationMatcher(BatchNorm2d) | NodeOperationMatcher(Conv2d)
    conv_node = NodeOperationMatcher(Conv2d) | NodeOperationMatcher(ConvTranspose2d)

    return bn_or_dw1x1_node, conv_node


def update_kernel_for_bn_folding_fn(conv_node: BaseNode,
                                    kernel: np.ndarray,
                                    weights_scale: np.ndarray) -> [np.ndarray, str]:
    """
    Args:
        conv_node: Convolution node to update the weight/kernel.
        kernel: The Convolution node's weight
        weights_scale: Weight scale factor in which to multiply the conv node's weight.

    Returns:
        The modified convolution node's weight/kernel/
    """
    if conv_node.is_match_type(ConvTranspose2d):
        _scale = weights_scale[None, :, None, None]
    else:
        _scale = weights_scale[:, None, None, None]
    if conv_node.is_match_type(ConvTranspose2d) and conv_node.framework_attr[GROUPS] > 1:
        # PyTorch ConvTranspose2d kernel with groups stacks groups on in_channels axis, so need to reshape the kernel
        # so the groups are stacked on the out_channels axis to match the scale vector (then reshape back to original
        # shape)
        _in_channels = int(conv_node.framework_attr[IN_CHANNELS]/conv_node.framework_attr[GROUPS])
        _out_channels = conv_node.framework_attr[OUT_CHANNELS]
        return (kernel.reshape((_in_channels, _out_channels, -1, 1)) * _scale).reshape(kernel.shape), KERNEL
    else:
        return kernel * _scale, KERNEL


def update_weights_for_bn_forward_folding_fn(conv_node: BaseNode,
                                             kernel: np.ndarray,
                                             bias: np.ndarray,
                                             weights_scale: np.ndarray,
                                             bias_factor: np.ndarray) -> [np.ndarray, np.ndarray, str]:
    """
    Args:
        conv_node: Convolution node to update the weight/kernel.
        kernel: The Convolution node's weight
        bias: The Convolution node's bias
        weights_scale: Weight scale factor in which to multiply the conv node's weight.
        bias_factor: factor for kernel to update the bias with: (beta - moving_mean * weights_scale)

    Returns:
        The modified convolution node's weight/kernel/
    """
    if conv_node.is_match_type(Conv2d) and conv_node.framework_attr[GROUPS] > 1:
        bias_update = (kernel * bias_factor[:, None, None, None]).flatten()
        _scale = weights_scale[:, None, None, None]
    elif conv_node.is_match_type(ConvTranspose2d):
        bias_update = (kernel * bias_factor[:, None, None, None]).sum(axis=0).flatten()
        _scale = weights_scale[:, None, None, None]
    else:
        bias_update = (kernel * bias_factor[None, :, None, None]).sum(axis=1).flatten()
        _scale = weights_scale[None, :, None, None]
    return kernel * _scale, bias + bias_update, KERNEL


def get_kernel_hw_fn(kernel: np.ndarray) -> [int, int]:
    """
    Args:
        kernel: The Convolution node's weight

    Returns:
        kernel height & width shape
    """
    return kernel.shape[2:]


def is_group_conv_fn(node: BaseNode) -> bool:
    """
    Check whether the node is a group-convolution
    Args:
        node: The Convolution node

    Returns:
        True if the node is a group convolution, else False
    """
    return (node.is_match_type(Conv2d) or node.is_match_type(ConvTranspose2d)) and \
        node.framework_attr[GROUPS] not in [node.framework_attr[IN_CHANNELS], 1]


def get_foldable_node_type_and_validity_fn(node: BaseNode) -> [bool, bool]:
    """
    Check whether the node to forward fold is a valid dw-convolution node or a
    batch-normalization node
    Args:
        node: The node to fold

    Returns:
        is_bn: True if the node is a batch norm, else False
        is_dw_valid: True if the node is a dw-convolution valid for folding or a batch-norm node, else False
    """
    is_bn = node.is_match_type(BatchNorm2d)
    is_dw = node.is_match_type(Conv2d) and node.framework_attr[GROUPS] == node.framework_attr[IN_CHANNELS]
    is_dw_valid = is_dw and np.all(np.array(node.get_weights_by_keys(KERNEL).shape[2:]) == 1)
    return is_bn, is_dw_valid


def pytorch_batchnorm_folding() -> BatchNormalizationFolding:
    """

    Returns:
        A BatchNormalizationFolding initialized for Pytorch models.
    """
    bn_node, source_node = batchnorm_folding_node_matchers()
    return BatchNormalizationFolding(source_node,
                                     bn_node,
                                     update_kernel_for_bn_folding_fn,
                                     KERNEL,
                                     BIAS,
                                     GAMMA,
                                     BETA,
                                     MOVING_MEAN,
                                     MOVING_VARIANCE,
                                     EPSILON,
                                     USE_BIAS,
                                     layer_name_str=None,  # torch.nn.Modules don't have an attribute 'name'
                                     )


def pytorch_batchnorm_forward_folding() -> BatchNormalizationForwardFolding:
    """

    Returns:
        A BatchNormalizationForwardFolding initialized for Pytorch models.
    """
    bn_node, source_node = batchnorm_forward_folding_node_matchers()
    return BatchNormalizationForwardFolding(bn_node,
                                            source_node,
                                            update_weights_for_bn_forward_folding_fn,
                                            get_kernel_hw_fn,
                                            is_group_conv_fn,
                                            get_foldable_node_type_and_validity_fn,
                                            KERNEL,
                                            BIAS,
                                            GAMMA,
                                            BETA,
                                            MOVING_MEAN,
                                            MOVING_VARIANCE,
                                            EPSILON,
                                            USE_BIAS,
                                            layer_name_str=None,  # torch.nn.Modules don't have an attribute 'name'
                                            )
