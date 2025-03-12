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

from model_compression_toolkit.target_platform_capabilities import LayerFilterParams, FrameworkQuantizationCapabilities

from typing import Set, Optional, List, Dict, Any
from model_compression_toolkit.core.common import BaseNode
import copy


class FusingInfo:
    """
    A class to manage fusing information, mapping fused operation IDs to sets of nodes.
    Designed to be extensible and type-safe.
    """

    def __init__(self, fqc: FrameworkQuantizationCapabilities, fusing_data: Optional[Dict[str, List[BaseNode]]] = None):
        """
        Initialize the FusingInfo with an optional dictionary of fusing data.

        Args:
            fusing_data (Optional[Dict[str, Set[BaseNode]]]): A dictionary mapping
                fused operation IDs to sets of nodes. Defaults to an empty dict.
        """
        self.fqc = fqc
        self._fusing_data = fusing_data or {}
        self._node_to_fused_node_map: Dict[str, str] = {}
        if fusing_data:
            self._update_node_mapping()

    def _update_node_mapping(self) -> None:
        """
        Update the node-to-fused-node mapping based on the current fusing data.
        """
        self._node_to_fused_node_map.clear()
        for op_id, nodes in self._fusing_data.items():
            for node in nodes:
                self._node_to_fused_node_map[node.name] = op_id

    def add_fused_operation(self, op_id: str, nodes: List[BaseNode]) -> None:
        """
        Add a new fused operation with the given ID and set of nodes.

        Args:
            op_id (str): The identifier for the fused operation.
            nodes (Set[BaseNode]): The set of nodes that form the fused operation.

        Raises:
            ValueError: If the operation ID already exists.
        """
        if op_id in self._fusing_data:
            raise ValueError(f"Fused operation {op_id} already exists.")
        self._fusing_data[op_id] = nodes
        # Update the mapping for these nodes
        for node in nodes:
            self._node_to_fused_node_map[node.name] = op_id

    def remove_fused_operation(self, op_id: str) -> None:
        """
        Remove a fused operation by its ID.

        Args:
            op_id (str): The identifier for the fused operation to remove.

        Raises:
            ValueError: If the operation ID does not exist.
        """
        if op_id not in self._fusing_data:
            raise ValueError(f"Fused operation {op_id} does not exist.")
        # Remove nodes from the mapping
        nodes = self._fusing_data[op_id]
        for node in nodes:
            self._node_to_fused_node_map.pop(node.name, None)
        del self._fusing_data[op_id]

    def get_fused_node_name(self, node_name: str) -> Optional[str]:
        """
        Get the name of the fused node containing the given original node name.

        Args:
            node_name: The name of a node from the original graph.

        Returns:
            The name of the fused node containing this node, or None if not fused.
        """
        return self._node_to_fused_node_map.get(node_name)

    def get_node_to_fused_node_map(self) -> Dict[str, str]:
        """
        Retrieve a copy of the mapping from original node names to fused node names.

        Returns:
            A dictionary mapping each original node name to its fused node name.
        """
        return self._node_to_fused_node_map.copy()

    def get_fused_nodes(self, op_id: str) -> Optional[Set[BaseNode]]:
        """
        Retrieve the set of nodes for a given fused operation ID.

        Args:
            op_id (str): The identifier for the fused operation.

        Returns:
            Optional[Set[BaseNode]]: The set of nodes for the operation, or None if not found.
        """
        return self._fusing_data.get(op_id)

    def is_node_in_fused_op(self, node: BaseNode) -> bool:
        """
        Check if a node is part of any fused operation.

        Args:
            node (BaseNode): The node to check.

        Returns:
            bool: True if the node is in any fused operation, False otherwise.
        """
        return any(node in nodes for nodes in self._fusing_data.values())

    def get_all_fused_operations(self) -> Dict[str, List[BaseNode]]:
        """
        Retrieve a copy of all fused operations.

        Returns:
            Dict[str, Set[BaseNode]]: A copy of the fusing data.
        """
        return copy.deepcopy(self._fusing_data)

    @staticmethod
    def generate_fused_op_id(nodes: List[BaseNode]) -> str:
        id = 'FusedNode_' + '_'.join([node.name for node in nodes])
        return id

    def validate(self, graph) -> None:
        """
        Validate that the fusing information is consistent with the given graph.

        This method checks:
        1. All nodes in the fusing data exist in the graph.
        2. Each fused sequence forms a linear chain in the graph, meaning:
           - Each node (except the last) has exactly one successor, which is the next node in the sequence.
           - Each node (except the first) has exactly one incoming edge, which is from the previous node in the sequence.
        3. Each fused sequence matches one of the predefined fusing patterns.

        Args:
            graph: The graph to validate against, expected to have methods like
                   nodes(), get_next_nodes(node), and incoming_edges(node).

        Raises:
            ValueError: If any validation check fails.
        """
        graph_nodes = set(graph.nodes())

        for op_id, nodes in self._fusing_data.items():
            node_set = set(nodes)
            node_index = {node: i for i, node in enumerate(nodes)}  # Track order in fusion pattern

            # Check 1: Ensure all nodes exist in the graph
            for node in nodes:
                if node not in graph_nodes:
                    raise ValueError(f"Fused operation {op_id} contains node {node.name} not in the graph.")

            # Check 2: Verify the sequence respects the fusion pattern
            for i, node in enumerate(nodes):
                successors = graph.get_next_nodes(node)
                predecessors = graph.get_prev_nodes(node)

                if i == 0:
                    if len(successors) == 0:
                        raise ValueError(f"Node {node.name} must have at least one successor.")
                elif 1 <= i < (len(nodes) - 1):
                    if len(successors) == 0 or len(predecessors) == 0:
                        raise ValueError(f"Node {node.name} must have at least one predecessor and one successor.")
                else:
                    if len(predecessors) == 0:
                        raise ValueError(f"Node {node.name} must have at least one predecessor.")

                # Ensure all successors are within the fused operation and appear **later** in the order
                for succ in successors:
                    if succ in node_set:
                        if node_index[succ] <= i:
                            raise ValueError(
                                f"Fused operation {op_id} has an invalid sequence: node {succ.name} appears before {node.name}."
                            )
                    elif i<(len(nodes)-1):
                        raise ValueError(
                            f"Fused operation {op_id} contains an external successor {succ.name} from node {node.name}."
                        )

                # Ensure all predecessors are within the fused operation and appear **earlier** in the order
                for pred in predecessors:
                    if pred in node_set:
                        if node_index[pred] >= i:
                            raise ValueError(
                                f"Fused operation {op_id} has an invalid sequence: node {pred.name} appears after {node.name}."
                            )
                    elif i>0 and pred not in graph.get_prev_nodes(nodes[0]):
                        raise ValueError(
                            f"Fused operation {op_id} contains an external predecessor {pred.name} to node {node.name}."
                        )

        # for op_id, nodes in self._fusing_data.items():
        #     # Check 1: Ensure all nodes exist in the graph
        #     for node in nodes:
        #         if node not in graph_nodes:
        #             raise ValueError(f"Fused operation {op_id} contains node {node.name} not in the graph.")
        #
        #     # Check 2: Verify the sequence forms a linear chain
        #     for i in range(len(nodes)):
        #         current_node = nodes[i]
        #         if i < (len(nodes)-1):
        #             next_node = nodes[i + 1]
        #
        #             successors = graph.get_next_nodes(current_node)
        #             if len(successors) != 1:
        #                 raise ValueError(
        #                     f"Fused operation {op_id} does not form a linear chain. Every node must have a single successor but {current_node.name} successors are: {successors}."
        #                 )
        #             if successors[0] != next_node:
        #                 raise ValueError(
        #                     f"Fused operation {op_id} does not form a linear chain. Expected to find node {next_node.name}, but found node {successors[0].name}"
        #                 )
        #         if i>0:
        #             prev_node_from_fusing_info = nodes[i - 1]
        #             prev_nodes_from_graph = graph.get_prev_nodes(current_node)
        #             if len(prev_nodes_from_graph) != 1:
        #                 raise ValueError(
        #                     f"Fused operation {op_id} does not form a linear chain. Node {current_node.name} must have exactly one incoming edge, but has {len(prev_nodes_from_graph)}."
        #                 )
        #             if prev_nodes_from_graph[0] != prev_node_from_fusing_info:
        #                 raise ValueError(
        #                     f"Fused operation {op_id} does not form a linear chain. Node {current_node.name} expected an incoming edge from {prev_node_from_fusing_info.name}, but found from {prev_nodes_from_graph[0].name}."
        #                 )

    def get_nodes_to_disable_act_quantization(self):
        res_nodes = []
        for nodes in self._fusing_data.values():
            res_nodes.extend(nodes[:-1])
        return res_nodes

    def is_nodes_eligible_to_be_fused(self, nodes: List[BaseNode]) -> bool:
        fusing_patterns = self.fqc.get_fusing_patterns()
        if not fusing_patterns:
            return False
        return is_valid_fusion(fusing_patterns=fusing_patterns, nodes=nodes)

    def __repr__(self) -> str:
        """
        Return a string representation of the fusing information.
        """
        fusing_data_repr = "\n".join(
            f"  {op_id}: [{', '.join(node.name for node in nodes)}]"
            for op_id, nodes in self._fusing_data.items()
        )
        mapping_repr = ", ".join(
            f"{node} -> {op_id}" for node, op_id in self._node_to_fused_node_map.items()
        )
        return (
            f"FusingInfo(\n"
            f"  Total fused operations: {len(self._fusing_data)}\n"
            f"  Fusing Data:\n{fusing_data_repr}\n"
            f"  Node-to-Fused Mapping:\n  {mapping_repr}\n"
            f")"
        )


class FusingInfoGenerator:
    def __init__(self, fqc):
        """
        Initialize the FusingInfoGenerator with a TPC object.

        Args:
            fqc: Target Platform Capabilities object providing fusing patterns.
        """
        self.fqc = fqc

    def generate_fusing_info(self, graph) -> FusingInfo:
        """
        Generate fusing information based on the graph and fusing patterns.

        Args:
            graph: The input graph to analyze, expected to have methods like
                   get_topo_sorted_nodes() and get_next_nodes(node).

        Returns:
            A dictionary where keys are unique fusion identifiers (e.g., 'fused_op_0')
            and values are lists of BaseNode objects representing nodes in that fusion.

        Notes:
            - Assumes filter_fusing_patterns and is_valid_fusion functions are defined elsewhere.
            - Nodes are processed in topological order to respect operation sequence.
            - Fusions are linear sequences (each node has exactly one successor).
            - Each node belongs to at most one fused operation.
        """
        # Retrieve fusing patterns from TPC
        fusing_patterns = self.fqc.get_fusing_patterns()
        if not fusing_patterns:
            return FusingInfo(fqc=self.fqc)

        # Determine the maximum length of fusing patterns
        max_layers_fusing = max(len(p) for p in fusing_patterns)

        # Get nodes in topological order
        nodes = graph.get_topo_sorted_nodes()

        # Initialize structures to track fusions
        fusing_info: Dict[str, List['BaseNode']] = {}
        fused_nodes = set()  # Tracks nodes already in a fusion

        # Process each node in topological order
        for node in nodes:
            if node in fused_nodes:
                continue  # Skip nodes already fused

            # Start with all possible patterns
            candidate_patterns = copy.deepcopy(fusing_patterns)
            current_sequence = []
            current_node = node

            # Try to build a sequence up to max_layers_fusing length
            for i in range(max_layers_fusing):
                # Filter patterns based on current node at position i
                candidate_patterns = filter_fusing_patterns(candidate_patterns, current_node, i)
                if not candidate_patterns:
                    break  # No patterns match, stop extending

                current_sequence.append(current_node)
                next_nodes = graph.get_next_nodes(current_node)

                # Ensure the sequence is linear (exactly one successor)
                if len(next_nodes) != 1:
                    break

                next_node = next_nodes[0]
                if next_node in fused_nodes:
                    break  # Avoid overlapping with existing fusions

                current_node = next_node

            # Check if the sequence forms a valid fusion
            if is_valid_fusion(fusing_patterns, current_sequence):
                # fused_op_id = f"fused_op_{fused_op_counter}"
                fused_op_id = FusingInfo.generate_fused_op_id(current_sequence)
                # fused_op_counter += 1
                assert fused_op_id not in fusing_info, f"{fused_op_id} is already in fusing info: {fusing_info}"
                fusing_info[fused_op_id] = current_sequence
                fused_nodes.update(current_sequence)

        return FusingInfo(fusing_data=fusing_info, fqc=self.fqc)


def filter_fusing_patterns(fusing_patterns: List[List[Any]], node: BaseNode, idx: int = 0) -> List[List[Any]]:
    """
    Update relevant fusing patterns object if layer number 'idx' inside the fusion matches the node
    Args:
        fusing_patterns: supported fusings
        node: node to decide if it can be a part of fusion
        idx: index of layer in the fusion
    Returns:
        fusing_patterns after filtering non-relevant fusions
    """
    valid_fusing_patterns = []
    for i, fusing_pattern in enumerate(fusing_patterns):
        if idx < len(fusing_pattern):
            if ((type(fusing_pattern[idx]) == LayerFilterParams and node.is_match_filter_params(fusing_pattern[idx])) or node.is_match_type(fusing_pattern[idx])):
                valid_fusing_patterns.append(fusing_pattern)

    # Return only valid patterns for this node
    return valid_fusing_patterns


def is_valid_fusion(fusing_patterns: List[List[Any]], nodes: List[BaseNode]) -> bool:
    """
    Check if the fusion is valid: exist in fusing_patterns
    Args:
        fusing_patterns: supported fusing patterns
        nodes: nodes which are participating in fusion
    Returns:
        whether the fusion in valid
    """
    fusion_depth = len(nodes)
    if fusion_depth <= 1:
        return False
    for fusing_pattern in fusing_patterns:
        if fusion_depth != len(fusing_pattern):
            continue
        counter = 0
        for i, layer in enumerate(fusing_pattern):
            if (type(layer) == LayerFilterParams and nodes[i].is_match_filter_params(layer)) or \
                    nodes[i].is_match_type(layer):
                counter += 1
        if counter == fusion_depth:
            return True
    return False
