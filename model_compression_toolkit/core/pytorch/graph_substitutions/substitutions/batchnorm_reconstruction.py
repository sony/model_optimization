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

from torch.nn import BatchNorm2d, Conv2d, ConvTranspose2d

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common.substitutions.batchnorm_reconstruction import \
    BatchNormalizationReconstruction
from model_compression_toolkit.core.pytorch.constants import GAMMA, BETA, MOVING_MEAN, MOVING_VARIANCE, \
    EPSILON, NUM_FEATURES, OUT_CHANNELS, EPSILON_VAL, MOMENTUM, MOMENTUM_VAL


def batchnorm_reconstruction_node_matchers():
    """
    Match linear layers for BN reconstruction.

    Returns:
        Node matcher for convolution type nodes.
    """
    source_node = NodeOperationMatcher(Conv2d) | \
                  NodeOperationMatcher(ConvTranspose2d)
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
                       framework_attr={NUM_FEATURES: source_node.framework_attr[OUT_CHANNELS],
                                       EPSILON: EPSILON_VAL,
                                       MOMENTUM: MOMENTUM_VAL},
                       input_shape=source_node.output_shape,
                       output_shape=source_node.output_shape,
                       weights=bn_node_weights,
                       layer_class=BatchNorm2d)
    return bn_node


def pytorch_batchnorm_reconstruction() -> BatchNormalizationReconstruction:
    """
    Reconstruct BatchNormalization after linear layers.
    Returns:
        A BatchNormalizationReconstruction initialized for Pytorch models.
    """
    source_node = batchnorm_reconstruction_node_matchers()
    return BatchNormalizationReconstruction(source_node, create_bn_node, GAMMA, BETA,
                                            MOVING_MEAN, MOVING_VARIANCE, EPSILON_VAL)
