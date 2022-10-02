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


from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.collectors.statistics_collector import BaseStatsCollector
from model_compression_toolkit.core.common.framework_info import FrameworkInfo

def create_stats_collector_for_node(node: common.BaseNode,
                                    fw_info: FrameworkInfo) -> BaseStatsCollector:
    """
    Gets a node and a groups list and create and return a statistics collector for a node
    according to whether its statistics should be collected and the prior information we
    have about this node.

    Args:
        node: Node to create its statistics collector.
        fw_info: Information relevant to a specific framework about what is out channel axis (for statistics per-channel).

    Returns:
        Statistics collector for statistics collection for the node.
    """

    if node.is_activation_quantization_enabled():
        min_output = getattr(node.prior_info, 'min_output', None)
        max_output = getattr(node.prior_info, 'max_output', None)
        stats_collector = common.StatsCollector(out_channel_axis=fw_info.out_channel_axis_mapping.get(node.type),
                                                init_min_value=min_output,
                                                init_max_value=max_output)
    else:
        stats_collector = common.NoStatsCollector()

    return stats_collector
