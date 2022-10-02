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
from tensorflow.keras.layers import BatchNormalization, DepthwiseConv2D, Conv2DTranspose, Conv2D

from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher, \
    NodeFrameworkAttrMatcher
from model_compression_toolkit.core.common.substitutions.batchnorm_refusing import BatchNormalizationRefusing
from model_compression_toolkit.core.keras.constants import KERNEL, LINEAR, ACTIVATION, BIAS, GAMMA, \
    BETA, \
    MOVING_MEAN, MOVING_VARIANCE, EPSILON, USE_BIAS, LAYER_NAME
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.batchnorm_folding import \
    update_kernel_for_bn_folding_fn


def batchnorm_refusing_node_matchers():
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


def keras_batchnorm_refusing() -> BatchNormalizationRefusing:
    """
    Re-fuse BatchNormalization into preceding linear layers.

    Returns:
        A BatchNormalizationRefusing initialized for Keras models.
    """
    bn_node, source_node = batchnorm_refusing_node_matchers()
    return BatchNormalizationRefusing(source_node, bn_node, update_kernel_for_bn_folding_fn, KERNEL, BIAS, GAMMA, BETA,
                                      MOVING_MEAN, MOVING_VARIANCE, EPSILON, USE_BIAS, LAYER_NAME)
