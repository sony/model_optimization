# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


from tensorflow.keras.layers import ReLU, Activation

from sony_model_optimization_package import common
from sony_model_optimization_package.common import FrameworkInfo
from sony_model_optimization_package.common.graph.node import Node
from sony_model_optimization_package.common.statistics_collector import BaseStatsContainer
from sony_model_optimization_package.keras.constants import LINEAR, ACTIVATION, RELU_MAX_VALUE, THRESHOLD, NEGATIVE_SLOPE


def get_stats_collector_for_activation_op(n: Node,
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


def get_stats_collector_for_kernel_op(n: common.Node,
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


def get_node_stats_collector(node: common.Node,
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
