# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
from model_compression_toolkit import FrameworkInfo
from model_compression_toolkit.common import BaseNode
from model_compression_toolkit.common.node_prior_info import NodePriorInfo


def create_node_prior_info(node: BaseNode,
                           fw_info: FrameworkInfo):
    """
    Create a NodePriorInfo object for a given node.

    Args:
        node: Node to create its prior info.
        fw_info: Information about a specific framework the node was generated from.

    Returns:
        NodePriorInfo object with info about the node.
    """

    min_output, max_output = None, None
    if fw_info.layers_has_min_max(node.type):
        min_output, max_output = fw_info.layer_min_max_mapping[node.type]
    return NodePriorInfo(min_output=min_output,
                         max_output=max_output)
