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
    from keras.src.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose

from model_compression_toolkit.core import QuantizationConfig
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
                  NodeOperationMatcher(DepthwiseConv2D) | \
                  NodeOperationMatcher(Conv2DTranspose)
    return linear_node


def keras_compute_activation_bias_correction_of_graph(graph: Graph,
                                                      quant_config: QuantizationConfig,
                                                      fw_impl: FrameworkImplementation) -> Graph:
    """
    Compute the activation bias correction term for graph based on a Keras model.

    Args:
        graph: Graph with nodes to compute the activation bias correction.
        quant_config: QuantizationConfig of how the model should be quantized.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

    Returns:
        Graph with activation bias correction term for each relevant node.
    """
    graph = compute_activation_bias_correction_of_graph(graph=graph,
                                                        quant_config=quant_config,
                                                        fw_impl=fw_impl,
                                                        activation_bias_correction_node_matchers=
                                                        activation_bias_correction_node_matchers,
                                                        kernel_size=KERNEL_SIZE)
    return graph
