#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

from typing import Dict, List

from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.graph.base_graph import OutTensor


class FusedLayerType:
    """
    Used to represent the type of fused layers, since __name__
    is accessed when the graph is displayed.
    """
    def __init__(self):
        self.__name__ = 'FusedLayer'
class GraphFuser:

    def create_fused_graph(self, graph: Graph) -> Dict[str, str]:
        """
        GraphFuser is responsible for fusing nodes in a networkx graph.
        The fusion process involves:
            1. Creating new fused nodes to represent these groups.
            2. Updating the graph structure to replace the original nodes with fused nodes.
            3. Maintaining mapping mapping of original node names to their fused node names.

        Args:
            graph: Graph to sue its nodes.

        Returns:
            Mapping of original node names to their fused node names
        """
        fused_nodes_mapping = {}
        # Iterate through each group of nodes to be fused
        for fused_nodes_list in graph.fused_nodes:
            new_fused_node = self._create_fused_node(fused_nodes_list)
            self._replace_nodes_with_fused_node(graph, fused_nodes_list, new_fused_node)
            # Update the mapping to keep track of which original nodes are now part of which fused nodes
            for node in fused_nodes_list:
                fused_nodes_mapping[node.name] = new_fused_node.name
        return fused_nodes_mapping

    def _create_fused_node(self, nodes: List[BaseNode]) -> BaseNode:
        """
        Create a new node that represents the fusion of the given nodes.

        Args:
            nodes: Nodes to create the fuse node that contain them.

        Returns:
            Node that represents the nodes to be fused.
        """
        # Create a new node with a name that reflects its components
        # Use the input shape of the first node and output shape of the last node
        fused_node = BaseNode(name='FusedNode_' + '_'.join([node.name for node in nodes]),
                              framework_attr={},
                              input_shape=nodes[0].input_shape,
                              output_shape=nodes[-1].output_shape,
                              weights={},
                              layer_class=FusedLayerType)

        # Preserve the final activation quantization configuration
        # This is important for maintaining the correct behavior of the fused node
        fused_node.final_activation_quantization_cfg = nodes[-1].final_activation_quantization_cfg

        return fused_node

    def _replace_nodes_with_fused_node(self,
                                      graph: Graph,
                                      nodes_to_fuse: List[BaseNode],
                                      fused_node: BaseNode):
        """
        Replace the specified nodes in the graph with a new fused node.

        Args:
            graph: Graph to replace the nodes_to_fuse with fused_node
            nodes_to_fuse: List of nodes to replace with a new fused node.
            fused_node: Node to add instead of nodes in fused_node.

        """
        if not nodes_to_fuse:
            return

        first_node = nodes_to_fuse[0]
        last_node = nodes_to_fuse[-1]

        # Update incoming edges: Connect predecessors of the first node to the fused node
        for predecessor in graph.get_prev_nodes(first_node):
            e_attr = graph.get_edge_data(predecessor, first_node)
            graph.add_edge(predecessor, fused_node, **(e_attr[0]))
            graph.remove_edge(predecessor, first_node)

        # Update outgoing edges: Connect the fused node to successors of the last node
        for successor in graph.get_next_nodes(last_node):
            e_attr = graph.get_edge_data(last_node, successor)
            graph.add_edge(fused_node, successor, **(e_attr[0]))
            graph.remove_edge(last_node, successor)

        # Remove internal edges between fused nodes
        # This step is necessary to maintain graph consistency
        for current_node in nodes_to_fuse[:-1]:
            subsequent_nodes = graph.get_next_nodes(current_node)
            for next_node in subsequent_nodes:
                assert next_node in nodes_to_fuse  # Ensure we're not removing edges outside the fusion
                graph.remove_edge(current_node, next_node)

        # Handle the case where fused nodes are part of the graph's outputs
        graph_output_tensors = graph.get_outputs()
        graph_output_nodes = [ot.node for ot in graph_output_tensors]
        for node in nodes_to_fuse:
            if node in graph_output_nodes:
                # If a fused node was an output, update the graph's outputs to use the new fused node
                node_to_remove_index = graph_output_nodes.index(node)
                graph_output_tensors[node_to_remove_index] = OutTensor(node=fused_node,
                                                                       node_out_index=graph_output_tensors[
                                                                           node_to_remove_index].node_out_index)
                graph.remove_node(node, new_graph_outputs=graph_output_tensors)
            else:
                # Remove the original node from the graph
                graph.remove_node(node)

        # Finally, add the new fused node to the graph
        graph.add_node(fused_node)
