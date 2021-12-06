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


from typing import Any, Dict

from model_compression_toolkit.common.graph.base_node import BaseNode

# Edge attributes:
EDGE_SOURCE_INDEX = 'source_index'
EDGE_SINK_INDEX = 'sink_index'


class Edge(object):
    """
    Edge in a directed graph.
    """

    def __init__(self,
                 source_node: BaseNode,
                 sink_node: BaseNode,
                 source_index: int,
                 sink_index: int):
        """
        Construct an edge by two nodes and indices.
        Args:
            source_node: Source node of the edge.
            sink_node: Sink node of the edge.
            source_index: Index of the source node of the edge.
            sink_index: Index of the sink node of the edge.
        """
        self.source_node = source_node
        self.sink_node = sink_node
        self.source_index = source_index
        self.sink_index = sink_index

    def get_attributes(self) -> Dict[str, Any]:
        """
        Get edge's attributes as a dictionary mapping an attribute name to its edge value.
        Returns: Edge's attributes.
        """

        return {EDGE_SOURCE_INDEX: self.source_index,
                EDGE_SINK_INDEX: self.sink_index}

    def __eq__(self, other: Any) -> bool:
        """
        Check if this edge is identical to the edge that was passed.
        Args:
            other: An object to compare to this edge.

        Returns:
            Whether the passed object is identical to this edge.
        """

        if isinstance(other, Edge):
            return other.sink_node == self.sink_node and \
                   other.source_node == self.source_node and \
                   other.source_index == self.source_index and \
                   other.sink_index == self.source_index

        return False

    def __repr__(self) -> str:
        """
        A string representing this edge for display.
        """

        return f'{self.source_node.name}:{self.source_index} -> ' \
               f'{self.sink_node.name}:{self.sink_index}'


def convert_to_edge(edge: Any) -> Edge:
    """
    Get an edge in the graph either as an Edge object (source node, destination node,
    source index and destination index) or as a networkx edge with edge data (triplet
    tuple with nodes and edge data) and return it as an Edge object.

    Args:
        edge: Edge in a graph to convert.

    Returns:
        An Edge object containing source node, destination node,
        source index and destination index.
    """

    if isinstance(edge, tuple) and len(edge) == 3:  # networkx edge representation with edge data
        src_node = edge[0]
        dst_node = edge[1]
        edge_data = edge[2]
        return Edge(src_node,
                    dst_node,
                    edge_data[EDGE_SOURCE_INDEX],
                    edge_data[EDGE_SINK_INDEX])

    elif isinstance(edge, Edge):  # it's already an Edge and no change need to be done
        return edge

    raise Exception('Edges list contains an object that is not a known edge format.')
