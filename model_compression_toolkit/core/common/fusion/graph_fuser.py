#  Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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

import copy
from typing import List

from model_compression_toolkit.core.common.fusion.fusing_info import FusingInfoGenerator
from model_compression_toolkit.core.common.fusion.fusing_metadata_wrapper import FusingMetadataWrapper
from model_compression_toolkit.core.common.graph.base_graph import Graph, BaseNode, OutTensor
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import CandidateNodeQuantizationConfig
from itertools import product


class FusedLayerType:
    """
    Used to represent the type of fused layers, since __name__
    is accessed when the graph is displayed.
    """
    def __init__(self):
        self.__name__ = 'FusedLayer'

class GraphFuser:
    def apply_node_fusion(self, wrapper: FusingMetadataWrapper) -> Graph:
        """
        Applies node fusion to the graph contained in the given FusingMetadataWrapper.

        The fusion process includes:
            1. Generating new fused nodes to replace groups of original nodes.
            2. Updating the graph structure to replace those nodes with the fused representations.

        Args:
            wrapper: A wrapper object that contains the graph and its fusing metadata.

        Returns:
            The updated graph with fused nodes replacing the original node groups.
        """
        wrapper_copy = copy.deepcopy(wrapper)
        expected_fusing_info = FusingInfoGenerator(wrapper.get_fusing_info().fqc).generate_fusing_info(
            wrapper.get_internal_graph())

        if expected_fusing_info != wrapper_copy.get_fusing_info():
            raise ValueError(
                f"Mismatch between expected and existing fusing information.\n"
                f"Expected:\n{expected_fusing_info}\nExisting:\n{wrapper_copy.get_fusing_info()}"
            )

        for fused_node_id, original_nodes in wrapper_copy.get_fusing_info().get_all_fused_operations().items():
            fused_node = self._create_fused_node(fused_node_id,
                                                 original_nodes)
            self._replace_nodes_with_fused_node(wrapper_copy.get_internal_graph(),
                                                original_nodes,
                                                fused_node)

        return wrapper_copy.get_internal_graph()


    @staticmethod
    def _create_fused_node(fused_node_id: str, nodes: List[BaseNode]) -> BaseNode:
        """
        Create a new node that represents the fusion of the given nodes.

        Args:
            nodes: Nodes to create the fuse node that contain them.

        Returns:
            Node that represents the nodes to be fused.
        """
        # Create a new node with a name that reflects its components
        # Use the input shape of the first node and output shape of the last node
        # TODO: consider replacing the fused node with a sub-model to allow inference on it, etc.
        fused_node = BaseNode(name=fused_node_id,
                              framework_attr={},
                              input_shape=nodes[0].input_shape,
                              output_shape=nodes[-1].output_shape,
                              weights={}, # TODO: update with weights of all nodes
                              layer_class=FusedLayerType)

        # Create candidates for this node (we assume that the weights configuration should be taken from the first node, and the activaion configuration
        # is the output quantization configuration of the last node. We ignore all configurations of middle nodes.
        weight_cfgs = [c.weights_quantization_cfg for c in nodes[0].candidates_quantization_cfg]
        activation_cfgs = [c.activation_quantization_cfg for c in nodes[-1].candidates_quantization_cfg]
        if weight_cfgs and activation_cfgs:
            combinations = list(product(weight_cfgs, activation_cfgs))
            fused_node.candidates_quantization_cfg = [
                CandidateNodeQuantizationConfig(weights_quantization_cfg=w, activation_quantization_cfg=a)
                for w, a in combinations
            ]

        # Keep the final configurations if they were set already.
        fused_node.final_weights_quantization_cfg = nodes[0].final_weights_quantization_cfg
        fused_node.final_activation_quantization_cfg = nodes[-1].final_activation_quantization_cfg

        return fused_node

    @staticmethod
    def _replace_nodes_with_fused_node(graph: Graph,
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


