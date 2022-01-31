# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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

from model_compression_toolkit.common import BaseNode
from model_compression_toolkit.common.graph.graph_matchers import NodeOperationMatcher, \
    NodeFrameworkAttrMatcher
from model_compression_toolkit.common.substitutions.batchnorm_folding import BatchNormalizationFolding
from model_compression_toolkit.keras.constants import KERNEL, LINEAR, ACTIVATION, DEPTHWISE_KERNEL, BIAS, GAMMA, BETA, \
    MOVING_MEAN, MOVING_VARIANCE, EPSILON, USE_BIAS, LAYER_NAME


def batchnorm_folding_node_matchers():
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


def update_kernel_for_bn_folding_fn(conv_node: BaseNode,
                                    kernel: np.ndarray,
                                    weights_scale):
    """
    Args:
        conv_node: Convolution node to update the weight/kernel.
        kernel: The Convolution node's weight
        weights_scale: Weight scale factor in which to multiply the conv node's weight.

    Returns:
        The modified convolution node's weight/kernel/
    """
    if conv_node.type == DepthwiseConv2D:
        kernel = kernel * weights_scale.reshape(1, 1, kernel.shape[-2], kernel.shape[-1])
    elif conv_node.type == Conv2DTranspose:
        kernel = kernel * weights_scale.reshape(1, 1, -1, 1)
    else:
        kernel = kernel * weights_scale.reshape(1, 1, 1, -1)

    if conv_node.type == DepthwiseConv2D:
        kernel_name = DEPTHWISE_KERNEL
    else:
        kernel_name = KERNEL

    return kernel, kernel_name


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
