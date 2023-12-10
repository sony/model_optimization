# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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

import keras.layers

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.pruning.pruning_section import PruningSection


def is_keras_entry_node(node: BaseNode):
    """

    Args:
        node:

    Returns:

    """
    return _is_keras_node_pruning_section_edge(node)


def is_keras_exit_node(node: BaseNode, match_entry_node: BaseNode, fw_info:FrameworkInfo):
    """

    Args:
        node:

    Returns:

    """
    return _is_keras_node_pruning_section_edge(node) and PruningSection.has_matching_channel_count(node,
                                                                                                   match_entry_node,
                                                                                                   fw_info)


def is_keras_node_intermediate_pruning_section(node: BaseNode):
    """

    Args:
        node:

    Returns:

    """
    # Nodes that are not Conv2D, Conv2DTranspose, DepthwiseConv2D, or Dense are considered intermediate.
    return node.type not in [keras.layers.DepthwiseConv2D,
                             keras.layers.Conv2D,
                             keras.layers.Conv2DTranspose,
                             keras.layers.Dense]


# Check if a Keras node is an edge of a pruning section.
def _is_keras_node_pruning_section_edge(node: BaseNode):
    """

    Args:
        node:

    Returns:

    """
    # Convolution nodes with group=1 or Dense layers are considered edges for pruning sections.
    if node.type in [keras.layers.Conv2D, keras.layers.Conv2DTranspose]:
        return node.framework_attr['groups'] == 1
    return node.type == keras.layers.Dense