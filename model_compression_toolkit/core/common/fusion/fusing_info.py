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
from dataclasses import dataclass, field

from typing import Optional, List, Dict, Any, Tuple
from model_compression_toolkit.core.common import BaseNode
import copy

# The prefix of each fused operator (the suffix is a combination of the
# nodes names that combine the fused operator).
FUSED_OP_ID_PREFIX = "FusedNode_"


@dataclass
class FusingInfo:
    """
    This class manages information about fused operations in a graph.

    The key responsibility of this class is maintaining a mapping between original nodes
    and their corresponding fused operation IDs. This mapping helps track which nodes
    belong to fused operations and validate this info is correct after changes in the graph.

    The core structures maintained are:
    - `fusing_data`: A dictionary mapping fused operation IDs to lists of nodes that belong to that operation.
    - `node_to_fused_node_map`: A dictionary mapping each node name to the ID of the fused operation it belongs to.

    """
    fusing_patterns: any
    fusing_data: Dict[str, Tuple['BaseNode']] = field(default_factory=dict)
    node_to_fused_node_map: Dict[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Validates and initializes mappings after dataclass instantiation."""
        for op_id, op_nodes in self.fusing_data.items():
            assert isinstance(op_id, str) and op_id.startswith(FUSED_OP_ID_PREFIX), f"Found invalid fused op id: {op_id}"
            assert isinstance(op_nodes, tuple) and len(op_nodes) > 1 and all(isinstance(n, BaseNode) for n in op_nodes), f"Found invalid fused op nodes: {op_nodes}"

        self._init_node_mapping()

    def _init_node_mapping(self) -> None:
        """
        Init the node-to-fused-node mapping based on the initial fusing data.
        """
        self.node_to_fused_node_map.clear()
        for op_id, nodes in self.fusing_data.items():
            for node in nodes:
                self.node_to_fused_node_map[node.name] = op_id

    def add_fused_operation(self, op_id: str, nodes: Tuple[BaseNode]) -> None:
        """
        Add a new fused operation with the given ID and set of nodes.

        Args:
            op_id (str): The identifier for the fused operation.
            nodes (Tuple[BaseNode]): The tuple of nodes that form the fused operation.

        Raises:
            ValueError: If the operation ID already exists.
        """
        if op_id in self.fusing_data:
            raise ValueError(f"Fused operation {op_id} already exists.")
        assert isinstance(nodes, tuple), f"Expected nodes to be a tuple but its type is {type(nodes)}"
        self.fusing_data[op_id] = nodes
        # Update the mapping for these nodes
        for node in nodes:
            self.node_to_fused_node_map[node.name] = op_id

    def remove_fused_operation(self, op_id: str) -> None:
        """
        Remove a fused operation by its ID.

        Args:
            op_id (str): The identifier for the fused operation to remove.

        Raises:
            ValueError: If the operation ID does not exist.
        """
        if op_id not in self.fusing_data:
            raise ValueError(f"Fused operation {op_id} does not exist.")
        # Remove nodes from the mapping
        nodes = self.fusing_data[op_id]
        for node in nodes:
            self.node_to_fused_node_map.pop(node.name, None)
        del self.fusing_data[op_id]

    def get_fused_node_name(self, node_name: str) -> Optional[str]:
        """
        Get the name of the fused node containing the given original node name.

        Args:
            node_name: The name of a node from the original graph.

        Returns:
            The name of the fused node containing this node, or None if not fused.
        """
        return self.node_to_fused_node_map.get(node_name)

    def get_node_to_fused_node_map(self) -> Dict[str, str]:
        """
        Retrieve a copy of the mapping from original node names to fused node names.

        Returns:
            A dictionary mapping each original node name to its fused node name.
        """
        return self.node_to_fused_node_map.copy()

    def get_fused_nodes(self, op_id: str) -> Optional[List[BaseNode]]:
        """
        Retrieve the list of nodes for a given fused operation ID.

        Args:
            op_id (str): The identifier for the fused operation.

        Returns:
            Optional[List[BaseNode]]: The list of nodes for the operation, or None if not found.
        """
        return self.fusing_data.get(op_id)

    def is_node_in_fused_op(self, node: BaseNode) -> bool:
        """
        Check if a node is part of any fused operation.

        Args:
            node (BaseNode): The node to check.

        Returns:
            bool: True if the node is in any fused operation, False otherwise.
        """
        return any(node in nodes for nodes in self.fusing_data.values())

    def get_all_fused_operations(self) -> Dict[str, Tuple[BaseNode]]:
        """
        Retrieve fused information.

        Returns:
            Dict[str, List[BaseNode]]: The fusing data.
        """
        return self.fusing_data


    @staticmethod
    def generate_fused_op_id(nodes: List[BaseNode]) -> str:
        """
        Generates an identifier for a fused operation by concatenating
        the names of the given nodes with a prefix.

        Args:
            nodes (List[BaseNode]): A list of nodes to be fused.

        Returns:
            str: An identifier string for the fused operation.
        """
        id = FUSED_OP_ID_PREFIX + '_'.join([node.name for node in nodes])
        return id

    def validate(self, graph) -> None:
        """
        Validate that the fusing information is consistent with the given graph.

        This method performs the following checks:

        1. **All nodes in the fusing data exist in the graph.**
        2. **Each fused sequence forms a valid linear chain in the graph:**
           - Each node (except the last) has exactly one successor, which is the next node in the sequence.
           - Each node (except the first) has exactly one predecessor, which is the previous node in the sequence.
        3. **Each fused sequence does not have unexpected external connections:**
           - No node in the sequence should have an external successor unless it is the last node.
           - No node in the sequence should have an external predecessor unless it is the first node.

        Args:
            graph: The computational graph to validate against. It is expected to have:
                   - `nodes()`: Returns a set of all nodes.
                   - `get_next_nodes(node)`: Returns a list of direct successor nodes.
                   - `get_prev_nodes(node)`: Returns a list of direct predecessor nodes.

        Raises:
            ValueError: If any validation check fails.
        """
        graph_nodes = set(graph.get_topo_sorted_nodes())  # Retrieve all nodes from the graph

        for op_id, nodes in self.fusing_data.items():
            node_set = set(nodes)  # Set of nodes in the current fusion sequence
            node_index = {node: i for i, node in enumerate(nodes)}  # Map node to its position in sequence

            # Check 1: Ensure all fused nodes exist in the graph
            for node in nodes:
                if node not in graph_nodes:
                    raise ValueError(f"Fused operation {op_id} contains node {node.name} not present in the graph.")

            # Check 2: Validate the fusion sequence forms a valid linear chain
            for i, node in enumerate(nodes):
                successors = graph.get_next_nodes(node)  # Direct successors of the current node
                predecessors = graph.get_prev_nodes(node)  # Direct predecessors of the current node

                # First node in the sequence: Must have at least one successor
                if i == 0:
                    if len(successors) == 0:
                        raise ValueError(f"First node {node.name} in fused operation {op_id} must have a successor.")

                # Middle nodes: Must have both a predecessor and a successor
                elif 1 <= i < (len(nodes) - 1):
                    if len(predecessors) == 0 or len(successors) == 0:
                        raise ValueError(
                            f"Node {node.name} in fused operation {op_id} must have both a predecessor and a successor.")

                # Last node in the sequence: Must have at least one predecessor
                else:
                    if len(predecessors) == 0:
                        raise ValueError(f"Last node {node.name} in fused operation {op_id} must have a predecessor.")

                # Check 3: Ensure successor relationships are within the fused sequence
                for succ in successors:
                    if succ in node_set:  # If successor is within fusion sequence, it must appear later
                        if node_index[succ] <= i:
                            raise ValueError(
                                f"Fused operation {op_id} has an invalid sequence: "
                                f"node {succ.name} appears before {node.name}."
                            )
                    elif i < (len(nodes) - 1):  # If not last node, external successor is not allowed
                        raise ValueError(
                            f"Fused operation {op_id} contains an external successor {succ.name} "
                            f"from node {node.name}, which is invalid."
                        )

                # Check 4: Ensure predecessor relationships are within the fused sequence
                for pred in predecessors:
                    if pred in node_set:  # If predecessor is within fusion sequence, it must appear earlier
                        if node_index[pred] >= i:
                            raise ValueError(
                                f"Fused operation {op_id} has an invalid sequence: "
                                f"node {pred.name} appears after {node.name}."
                            )
                    elif i > 0 and pred not in graph.get_prev_nodes(
                            nodes[0]):  # External predecessors allowed only for first node
                        raise ValueError(
                            f"Fused operation {op_id} contains an external predecessor {pred.name} "
                            f"to node {node.name}, which is invalid."
                        )

    def is_nodes_eligible_to_be_fused(self, nodes: List[BaseNode]) -> bool:
        """
        Check whether the given nodes are eligible to be fused based on predefined fusing patterns.

        This method retrieves the fusing patterns from `self.fqc` and verifies whether the
        given sequence of nodes matches any of the valid patterns.

        Args:
            nodes (List[BaseNode]): The list of nodes to check for fusion eligibility.

        Returns:
            bool: True if the nodes can be fused according to fusing patterns, otherwise False.
        """
        # If no fusing patterns are defined, fusion is not possible
        if not self._fusing_patterns:
            return False

        # Check if the provided nodes match a valid fusion pattern
        return is_valid_fusion(fusing_patterns=self._fusing_patterns, nodes=nodes)


    def __repr__(self) -> str:
        """
        Return a string representation of the fusing information.
        """
        fusing_data_repr = "\n".join(
            f"  {op_id}: [{', '.join(node.name for node in nodes)}]"
            for op_id, nodes in self.fusing_data.items()
        )
        mapping_repr = ", ".join(
            f"{node} -> {op_id}" for node, op_id in self.node_to_fused_node_map.items()
        )
        return (
            f"FusingInfo(\n"
            f"  Total fused operations: {len(self.fusing_data)}\n"
            f"  Fusing Data:\n{fusing_data_repr}\n"
            f"  Node-to-Fused Mapping:\n  {mapping_repr}\n"
            f")"
        )


class FusingInfoGenerator:
    def __init__(self, fusing_patterns):
        self._fusing_patterns = fusing_patterns

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
            - Assumes get_valid_fusing_patterns_for_node and is_valid_fusion functions are defined elsewhere.
            - Nodes are processed in topological order to respect operation sequence.
            - Fusions are linear sequences (each node has exactly one successor).
            - Each node belongs to at most one fused operation.
        """
        if not self._fusing_patterns:
            return FusingInfo(fusing_patterns=self._fusing_patterns)

        # Determine the maximum length of fusing patterns
        max_layers_fusing = max(len(p) for p in self._fusing_patterns)

        # Get nodes in topological order
        nodes = graph.get_topo_sorted_nodes()

        # Initialize structures to track fusions
        fusing_info: Dict[str, Tuple['BaseNode']] = {}
        fused_nodes = set()  # Tracks nodes already in a fusion

        # Process each node in topological order
        for node in nodes:
            if node in fused_nodes:
                continue  # Skip nodes already fused

            # Start with all possible patterns
            candidate_patterns = copy.deepcopy(self._fusing_patterns)
            current_sequence = []
            current_node = node

            # Try to build a sequence up to max_layers_fusing length
            for i in range(max_layers_fusing):
                # Filter patterns based on current node at position i
                candidate_patterns = get_valid_fusing_patterns_for_node(candidate_patterns, current_node, i)
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
            if is_valid_fusion(self._fusing_patterns, current_sequence):
                fused_op_id = FusingInfo.generate_fused_op_id(current_sequence)
                assert fused_op_id not in fusing_info, f"{fused_op_id} is already in fusing info: {fusing_info}"
                fusing_info[fused_op_id] = tuple(current_sequence)
                fused_nodes.update(current_sequence)

        return FusingInfo(fusing_data=fusing_info, fusing_patterns=self._fusing_patterns)


def get_valid_fusing_patterns_for_node(fusing_patterns: List[List[Any]],
                                       node: BaseNode,
                                       idx: int = 0) -> List[List[Any]]:
    """
    Returns only the fusing patterns where a specific layer (at index idx) matches the given node — either by type or filter params.

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
            if ((type(fusing_pattern[idx]) == LayerFilterParams and node.is_match_filter_params(
                    fusing_pattern[idx])) or node.is_match_type(fusing_pattern[idx])):
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
