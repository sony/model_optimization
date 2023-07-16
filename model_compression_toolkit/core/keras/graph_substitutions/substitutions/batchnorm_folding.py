# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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
from tensorflow.keras.layers import BatchNormalization, DepthwiseConv2D, Conv2DTranspose, Conv2D

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher, \
    NodeFrameworkAttrMatcher
from model_compression_toolkit.core.common.substitutions.batchnorm_folding import BatchNormalizationFolding, BatchNormalizationForwardFolding
from model_compression_toolkit.core.keras.constants import KERNEL, LINEAR, ACTIVATION, DEPTHWISE_KERNEL, BIAS, GAMMA, BETA, \
    MOVING_MEAN, MOVING_VARIANCE, EPSILON, USE_BIAS, LAYER_NAME, GROUPS


def batchnorm_folding_node_matchers() -> [BaseNode, BaseNode]:
    """
    Function generates matchers for matching:
    (DepthwiseConv2D, Conv2D, Conv2DTranspose)[activation=linear] -> BatchNormalization.

    Returns:
        Matcher for batch norm nodes, and source nodes.
    """
    bn_node = NodeOperationMatcher(BatchNormalization)
    conv_node = NodeOperationMatcher(DepthwiseConv2D) | \
                NodeOperationMatcher(Conv2D) | \
                NodeOperationMatcher(Conv2DTranspose)

    activation_linear = NodeFrameworkAttrMatcher(ACTIVATION, LINEAR)
    source_node = conv_node & activation_linear
    return bn_node, source_node


def batchnorm_forward_folding_node_matchers() -> [BaseNode, BaseNode]:
    """
    Function that generates matchers for matching:
    (BatchNormalization, DepthwiseConv2D 1x1) -> (DepthwiseConv2D, Conv2D, Conv2DTranspose).

    Returns:
        Matcher for batch norm nodes, and source nodes.
    """
    bn_or_dw1x1_node = NodeOperationMatcher(BatchNormalization) | NodeOperationMatcher(DepthwiseConv2D)
    conv_node = NodeOperationMatcher(DepthwiseConv2D) | \
                NodeOperationMatcher(Conv2D) | \
                NodeOperationMatcher(Conv2DTranspose)

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
    if conv_node.type == DepthwiseConv2D:
        kernel = kernel * weights_scale.reshape((1, 1, kernel.shape[-2], kernel.shape[-1]))
    elif conv_node.type == Conv2DTranspose:
        kernel = kernel * weights_scale.reshape((1, 1, -1, 1))
    else:
        kernel = kernel * weights_scale.reshape((1, 1, 1, -1))

    if conv_node.type == DepthwiseConv2D:
        kernel_name = DEPTHWISE_KERNEL
    else:
        kernel_name = KERNEL

    return kernel, kernel_name


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
    if conv_node.type == DepthwiseConv2D:
        bias_update = kernel * bias_factor.reshape((1, 1, -1, 1))
        kernel = kernel * weights_scale.reshape((1, 1, -1, 1))
    elif conv_node.type == Conv2DTranspose:
        bias_update = (kernel * bias_factor.reshape((1, 1, 1, -1))).sum(3)
        kernel = kernel * weights_scale.reshape((1, 1, 1, -1))
    else:
        bias_update = (kernel * bias_factor.reshape((1, 1, -1, 1))).sum(2)
        kernel = kernel * weights_scale.reshape((1, 1, -1, 1))

    if conv_node.type == DepthwiseConv2D:
        kernel_name = DEPTHWISE_KERNEL
    else:
        kernel_name = KERNEL

    return kernel, bias + bias_update.flatten(), kernel_name


def get_kernel_hw_fn(kernel: np.ndarray) -> [int, int]:
    """
    Args:
        kernel: The Convolution node's weight

    Returns:
        kernel height & width shape
    """
    return kernel.shape[:2]


def is_group_conv_fn(node: BaseNode) -> bool:
    """
    Check whether the node is a group-convolution
    Args:
        node: The Convolution node

    Returns:
        True if the node is a group convolution, else False
    """
    return (node.type == Conv2D) and node.framework_attr[GROUPS] > 1


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
    is_bn = node.type is BatchNormalization
    is_dw = node.type is DepthwiseConv2D
    is_dw_valid = is_dw and np.all(np.array(node.get_weights_by_keys(DEPTHWISE_KERNEL).shape[:2]) == 1)
    return is_bn, is_dw_valid


def keras_batchnorm_folding() -> BatchNormalizationFolding:
    """

    Returns:
        A BatchNormalizationFolding initialized for Keras models.
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
                                     LAYER_NAME)


def keras_batchnorm_forward_folding() -> BatchNormalizationForwardFolding:
    """

    Returns:
        A BatchNormalizationForwardFolding initialized for Keras models.
    """
    bn_node, conv_node = batchnorm_forward_folding_node_matchers()
    return BatchNormalizationForwardFolding(bn_node,
                                            conv_node,
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
                                            LAYER_NAME)
