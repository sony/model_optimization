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

import keras
import tensorflow as tf

from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.common.substitutions.remove_identity import remove_identity_node


class RemoveIdentity(common.BaseSubstitution):
    """
    Remove Identity layers from the graph.
    """

    def __init__(self):
        nodes = NodeOperationMatcher(keras.layers.Identity) | NodeOperationMatcher(tf.identity)
        super().__init__(matcher_instance=nodes)

    def substitute(self,
                   graph: Graph,
                   node: BaseNode) -> Graph:
        """
        The method to perform the substitution of the identity keras node by
        reconnecting its input directly to its output, effectively removing the node
        from the graph.

        Args:
            graph: The current graph of operations where the node resides.
            node: The specific `BaseNode` that is matched to be an Identity operation.

        Returns:
            Graph: The updated graph after removing the identity node.
        """
        return remove_identity_node(graph, node)

