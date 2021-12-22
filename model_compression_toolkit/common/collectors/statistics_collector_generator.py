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


from model_compression_toolkit import common
from model_compression_toolkit.common.collectors.statistics_collector import BaseStatsCollector


def create_stats_collector_for_node(node: common.BaseNode) -> BaseStatsCollector:
    """
    Gets a node and a groups list and create and return a statistics collector for a node
    according to whether its statistics should be collected and the prior information we
    have about this node.

    Args:
        node: Node to create its statistics collector.

    Returns:
        Statistics collector for statistics collection for the node.
    """

    if node.is_activation_quantization_enabled():
        stats_collector = common.StatsCollector(*(node.prior_info.get_min_max()))
    else:
        stats_collector = common.NoStatsCollector()

    return stats_collector
