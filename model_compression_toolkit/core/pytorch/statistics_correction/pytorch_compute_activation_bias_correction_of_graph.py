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

from torch.nn import Conv2d, Linear, ConvTranspose2d

from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common.statistics_correction.compute_activation_bias_correction_of_graph import \
    compute_activation_bias_correction_of_graph
from model_compression_toolkit.core.pytorch.constants import KERNEL_SIZE


def activation_bias_correction_node_matchers():
    # Match linear layers where we can add a correction.
    linear_node = NodeOperationMatcher(Linear) | NodeOperationMatcher(Conv2d) | NodeOperationMatcher(ConvTranspose2d)
    return linear_node


def pytorch_compute_activation_bias_correction_of_graph(graph: Graph,
                                                        quant_config: QuantizationConfig,
                                                        fw_info: FrameworkInfo,
                                                        fw_impl: FrameworkImplementation) -> Graph:
    """
    Compute the activation bias correction term for graph based on a PyTorch model.

    Args:
        graph: Graph with nodes to compute the activation bias correction.
        quant_config: QuantizationConfig of how the model should be quantized.
        fw_info: Framework info like lists of nodes their kernel should quantized.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

    Returns:
        Graph with activation bias correction term for each relevant node.
    """
    graph = compute_activation_bias_correction_of_graph(graph=graph,
                                                        quant_config=quant_config,
                                                        fw_info=fw_info,
                                                        fw_impl=fw_impl,
                                                        activation_bias_correction_node_matchers=
                                                        activation_bias_correction_node_matchers,
                                                        kernel_size=KERNEL_SIZE)
    return graph
