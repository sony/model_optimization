# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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

import tensorflow as tf
from packaging import version

from model_compression_toolkit.core.keras.constants import KERNEL_SIZE

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers import Conv2D, DepthwiseConv2D, Dense, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, Flatten, Cropping2D, Permute, GlobalAveragePooling2D
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, Flatten, Cropping2D, Permute, GlobalAveragePooling2D

from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common.statistics_correction.compute_activation_bias_correction_of_graph import \
    compute_activation_bias_correction_of_graph


def activation_bias_correction_node_matchers():
    # Match linear layers where we can add a correction.
    linear_node = NodeOperationMatcher(Conv2D) | \
                  NodeOperationMatcher(Dense) | \
                  NodeOperationMatcher(DepthwiseConv2D)

    # Match bypass layers what don't affect the quantization process.
    bypass_node = (NodeOperationMatcher(Cropping2D) |
                   NodeOperationMatcher(GlobalAveragePooling2D) |
                   NodeOperationMatcher(Dropout) |
                   NodeOperationMatcher(ZeroPadding2D) |
                   NodeOperationMatcher(MaxPooling2D) |
                   NodeOperationMatcher(tf.split) |
                   NodeOperationMatcher(tf.cast) |
                   NodeOperationMatcher(tf.unstack) |
                   NodeOperationMatcher(tf.__operators__.getitem) |
                   NodeOperationMatcher(tf.strided_slice) |
                   NodeOperationMatcher(Reshape) |
                   NodeOperationMatcher(tf.reshape) |
                   NodeOperationMatcher(Permute) |
                   NodeOperationMatcher(tf.transpose) |
                   NodeOperationMatcher(Flatten) |
                   NodeOperationMatcher(tf.gather) |
                   NodeOperationMatcher(tf.compat.v1.gather))

    return linear_node, bypass_node


def keras_compute_activation_bias_correction_of_graph(graph: Graph,
                                                      core_config: CoreConfig,
                                                      fw_info: FrameworkInfo,
                                                      fw_impl: FrameworkImplementation) -> Graph:
    """
    Compute the activation bias correction term for graph based on a Keras model.

    Args:
        graph: Graph with nodes to compute the activation bias correction.
        core_config: Configuration object containing parameters of how the model should be quantized.
        fw_info: Framework info like lists of nodes their kernel should quantized.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

    Returns:
        Graph with activation bias correction term for each relevant node.
    """
    graph = compute_activation_bias_correction_of_graph(graph=graph,
                                                        core_config=core_config,
                                                        fw_info=fw_info,
                                                        fw_impl=fw_impl,
                                                        activation_bias_correction_node_matchers=
                                                        activation_bias_correction_node_matchers,
                                                        kernel_size=KERNEL_SIZE)
    return graph
