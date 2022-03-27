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
from typing import Any, Tuple
import numpy as np
from torch.nn import BatchNorm2d

from model_compression_toolkit import FrameworkInfo
from model_compression_toolkit.common import BaseNode, Graph
from model_compression_toolkit.common.node_prior_info import NodePriorInfo
from model_compression_toolkit.pytorch.constants import MOVING_MEAN, MOVING_VARIANCE, GAMMA, BETA


def create_node_prior_info(node: BaseNode,
                           fw_info: FrameworkInfo,
                           graph: Graph):
    """
    Create a NodePriorInfo object for a given node.

    Args:
        node: Node to create its prior info.
        fw_info: Information about a specific framework the node was generated from.
        graph: Graph to check the next node type.

    Returns:
        NodePriorInfo object with info about the node.
    """

    min_output, max_output = None, None
    if fw_info.layers_has_min_max(node.type):
        min_output, max_output = fw_info.layer_min_max_mapping[node.type]
    mean_output, std_output = _get_mean_std_outputs(node=node,
                                                    graph=graph)
    return NodePriorInfo(min_output=min_output,
                         max_output=max_output,
                         mean_output=mean_output,
                         std_output=std_output)


def _get_mean_std_outputs(node: BaseNode,
                          graph: Graph) -> Tuple[Any, Any]:
    """
    Return the mean/std output values of a node if known.
    If one of them (or both of them) is unknown - return None instead of a value.
    Args:
        node: Node to create its prior info.
        graph: Graph to check the next node type.

    Returns:
        Mean/Std output values if known.
    """
    mean_output, std_output = None, None

    if node.type == BatchNorm2d:
        mean_output = node.get_weights_by_keys(BETA)
        if node.get_weights_by_keys(GAMMA) is None:
            std_output = 1.0
        else:
            std_output = np.abs(node.get_weights_by_keys(GAMMA))
        if mean_output is None:
            mean_output = 0.0
    else:
        next_node_list = graph.get_next_nodes(node)
        bn_nodes = [bn_node for bn_node in next_node_list if bn_node.type == BatchNorm2d]
        if len(bn_nodes) != 0:
            bn_node = bn_nodes[0]
            moving_variance = bn_node.get_weights_by_keys(MOVING_VARIANCE)
            std_output = np.sqrt(moving_variance)
            mean_output = bn_node.get_weights_by_keys(MOVING_MEAN)

    return mean_output, std_output
