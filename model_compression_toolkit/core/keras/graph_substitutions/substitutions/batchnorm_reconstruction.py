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
from typing import Dict, Any

from tensorflow.keras.layers import BatchNormalization, DepthwiseConv2D, Conv2DTranspose, Conv2D

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher, \
    NodeFrameworkAttrMatcher
from model_compression_toolkit.core.common.substitutions.batchnorm_reconstruction import \
    BatchNormalizationReconstruction
from model_compression_toolkit.core.keras.constants import LINEAR, ACTIVATION, GAMMA, \
    BETA, MOVING_MEAN, MOVING_VARIANCE, EPSILON, EPSILON_VAL, MOMENTUM, MOMENTUM_VAL


def batchnorm_reconstruction_node_matchers() -> NodeOperationMatcher:
    """
    Match linear layers for BN reconstruction.

    Returns:
        Node matcher for convolution type nodes.
    """
    conv_node = NodeOperationMatcher(DepthwiseConv2D) | \
                NodeOperationMatcher(Conv2D) | \
                NodeOperationMatcher(Conv2DTranspose)

    activation_linear = NodeFrameworkAttrMatcher(ACTIVATION, LINEAR)
    source_node = conv_node & activation_linear
    return source_node


def create_bn_node(source_node: BaseNode,
                   bn_node_weights: Dict[Any, Any]):
    """
    Creates BN nodes after linear nodes.
    Args:
        source_node: Node matcher for convolution type nodes.
        bn_node_weights: dictionary of BN weights.
    """
    bn_node = BaseNode(name=source_node.name + '_reconstructed',
                       framework_attr={EPSILON: EPSILON_VAL, MOMENTUM: MOMENTUM_VAL},
                       input_shape=source_node.output_shape,
                       output_shape=source_node.output_shape,
                       weights=bn_node_weights,
                       layer_class=BatchNormalization)
    return bn_node


def keras_batchnorm_reconstruction() -> BatchNormalizationReconstruction:
    """
    Reconstruct BatchNormalization after linear layers.
    Returns:
        A BatchNormalizationReconstruction initialized for keras models.
    """
    source_node = batchnorm_reconstruction_node_matchers()
    return BatchNormalizationReconstruction(source_node,
                                            create_bn_node,
                                            GAMMA,
                                            BETA,
                                            MOVING_MEAN,
                                            MOVING_VARIANCE,
                                            EPSILON_VAL)
