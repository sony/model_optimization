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


from tensorflow.keras.layers import ReLU, Activation

from model_compression_toolkit import common
from model_compression_toolkit.common import FrameworkInfo
from model_compression_toolkit.common.graph.base_node import BaseNode
from model_compression_toolkit.common.statistics_collector import BaseStatsContainer
from model_compression_toolkit.keras.constants import LINEAR, ACTIVATION, RELU_MAX_VALUE, THRESHOLD, NEGATIVE_SLOPE


def get_stats_collector_for_activation_op(n: BaseNode,
                                          fw_info: FrameworkInfo) -> common.StatsContainer:
    """
    Create and initial a statistics collector for an activation layer. If the activation function's min/max
    output values are known, the statistics collector is initialized with these values.

    Args:
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
        groups of layers by how they should be quantized, etc.)
        n: Node to create a statistics collector for it.

    Returns:
        Statistics collector initialized with known min/max values.
    """

    if n.layer_class == ReLU:
        negative_slope = n.framework_attr[NEGATIVE_SLOPE]
        threshold = n.framework_attr[THRESHOLD]
        init_max = n.framework_attr[RELU_MAX_VALUE]
        return common.StatsContainer(init_min_value=threshold if negative_slope == 0 else None,
                                     init_max_value=init_max)

    if n.layer_class == Activation:
        init_min, init_max = fw_info.activation_min_max_mapping[n.framework_attr[ACTIVATION]]
        return common.StatsContainer(init_min_value=init_min,
                                     init_max_value=init_max)

    if fw_info.layers_has_min_max(n.layer_class):
        init_min, init_max = fw_info.layer_min_max_mapping[n.layer_class]
        return common.StatsContainer(init_min_value=init_min,
                                     init_max_value=init_max)

    return common.StatsContainer()


def get_stats_collector_for_kernel_op(n: common.BaseNode,
                                      fw_info: FrameworkInfo) -> BaseStatsContainer:
    """
    Create and initial a statistics collector for a linear operator. If the layer has an activation function and
    its min/max output values are known, the statistics collector is initialized with these values.
    If the layer's output should not be quantized, NoStatsContainer is created.

    Args:
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
        groups of layers by how they should be quantized, etc.)
        n: Node to create a statistics collector for it.

    Returns:
        BaseStatsContainer according to statistics that are collected.
    """

    if n.framework_attr[ACTIVATION] == LINEAR and n.output_quantization:
        return common.StatsContainer()

    if n.framework_attr[ACTIVATION] == LINEAR and not n.output_quantization:
        return common.NoStatsContainer()

    if n.framework_attr[ACTIVATION] in fw_info.activation_min_max_mapping.keys():
        min_value, max_value = fw_info.activation_min_max_mapping[n.framework_attr[ACTIVATION]]
        return common.StatsContainer(init_min_value=min_value,
                                     init_max_value=max_value)

    return common.StatsContainer()


def get_node_stats_collector(node: common.BaseNode,
                             fw_info: common.FrameworkInfo) -> common.statistics_collector.BaseStatsContainer:
    """
    Gets a node and a groups list and create and return a statistics collector for the node
    according to the group the node is in.

    Args:
        node: Node to create its statistics collector.
        fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
        groups of layers by how they should be quantized, etc.)

    Returns:
        Statistics collector for statistics collection for the node.
    """

    stats_collector = get_stats_collector_for_activation_op(node, fw_info)
    if fw_info.in_no_quantization_ops(node):  # node should not be quantized
        stats_collector = common.NoStatsContainer()

    if fw_info.in_kernel_ops(node):  # node's kernel should be quantized
        stats_collector = get_stats_collector_for_kernel_op(node, fw_info)

    return stats_collector
