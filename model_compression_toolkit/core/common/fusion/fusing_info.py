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

from model_compression_toolkit.target_platform_capabilities import LayerFilterParams
from dataclasses import dataclass, field

from typing import Optional, List, Dict, Any, Tuple
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
    fusing_patterns: any = None
    fusing_data: Dict[str, Tuple['BaseNode']] = field(default_factory=dict)
    node_to_fused_node_map: Dict[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Validates and initializes mappings after dataclass instantiation."""
        for op_id, op_nodes in self.fusing_data.items():
            assert isinstance(op_id, str) and op_id.startswith(FUSED_OP_ID_PREFIX), f"Found invalid fused op id: {op_id}"
            assert isinstance(op_nodes, tuple) and len(op_nodes) > 1, f"Found invalid fused op nodes: {op_nodes}"

        self._init_node_mapping()

    def _init_node_mapping(self) -> None:
        """
        Init the node-to-fused-node mapping based on the initial fusing data.
        """
        self.node_to_fused_node_map.clear()
        for op_id, nodes in self.fusing_data.items():
            for node in nodes:
                self.node_to_fused_node_map[node.name] = op_id

    def add_fused_operation(self, op_id: str, nodes: Tuple['BaseNode']) -> None:
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

    def get_fused_nodes(self, op_id: str) -> Optional[List['BaseNode']]:
        """
        Retrieve the list of nodes for a given fused operation ID.

        Args:
            op_id (str): The identifier for the fused operation.

        Returns:
            Optional[List[BaseNode]]: The list of nodes for the operation, or None if not found.
        """
        return self.fusing_data.get(op_id)

    def is_node_in_fused_op(self, node: 'BaseNode') -> bool:
        """
        Check if a node is part of any fused operation.

        Args:
            node (BaseNode): The node to check.

        Returns:
            bool: True if the node is in any fused operation, False otherwise.
        """
        return any(node in nodes for nodes in self.fusing_data.values())

    def get_all_fused_operations(self) -> Dict[str, Tuple['BaseNode']]:
        """
        Retrieve fused information.

        Returns:
            Dict[str, List[BaseNode]]: The fusing data.
        """
        return self.fusing_data

    @staticmethod
    def generate_fused_op_id(nodes: List['BaseNode']) -> str:
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

    def validate(self, graph: 'Graph') -> None:
        """
        Validate that the fusing information is consistent with the given graph and generation logic.

        This method performs the following checks:
        1. All nodes in the fusing data exist in the graph.
        2. Each fused sequence forms a valid linear chain in the graph:
           - Each node (except the last) has exactly one successor, which is the next node in the sequence.
        3. No node is part of more than one fused operation.
        4. Each fused sequence matches a valid fusing pattern from the original set.

        Args:
            graph: The computational graph to validate against. It is expected to have:
                   - `get_topo_sorted_nodes()`: Returns a list of nodes in topological order.
                   - `get_next_nodes(node)`: Returns a list of direct successor nodes.

        Raises:
            ValueError: If any validation check fails.
        """
        graph_nodes = set(graph.get_topo_sorted_nodes())  # Retrieve all nodes from the graph
        all_fused_nodes = set()  # Track all nodes used in fusions to ensure no overlap

        for op_id, nodes in self.fusing_data.items():
            # Check 1: Ensure all fused nodes exist in the graph
            for node in nodes:
                if node not in graph_nodes:
                    raise ValueError(f"Fused operation {op_id} contains node {node.name} not present in the graph.")

            # Check 2: Validate the fusion sequence forms a valid linear chain
            for i in range(len(nodes) - 1):  # Up to the second-to-last node
                current_node = nodes[i]
                next_node = nodes[i + 1]
                successors = graph.get_next_nodes(current_node)
                if len(successors) != 1 or successors[0] != next_node:
                    raise ValueError(
                        f"Fused operation {op_id} is not a valid linear chain: "
                        f"node {current_node.name} does not connect directly to {next_node.name} "
                        f"with exactly one successor (found successors: {[n.name for n in successors]})."
                    )

            # Check 3: Ensure no node is reused across fusions
            node_set = set(nodes)
            overlap = node_set & all_fused_nodes
            if overlap:
                raise ValueError(
                    f"Fused operation {op_id} contains nodes already used in another fusion: "
                    f"{[node.name for node in overlap]}."
                )
            all_fused_nodes.update(node_set)

            # Check 4: Ensure the sequence matches a valid fusing pattern
            if not is_valid_fusion(self.fusing_patterns, nodes):
                raise ValueError(
                    f"Fused operation {op_id} does not match any valid fusing pattern "
                    f"from {self.fusing_patterns}."
                )

    def is_nodes_eligible_to_be_fused(self, nodes: List['BaseNode']) -> bool:
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
        if not self.fusing_patterns:
            return False

        # Check if the provided nodes match a valid fusion pattern
        return is_valid_fusion(fusing_patterns=self.fusing_patterns, nodes=nodes)

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

    def generate_fusing_info(self, graph: 'Graph') -> FusingInfo:
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

        # Find max fusion
        max_layers_fusing = max([len(fusing_pattern) for fusing_pattern in self._fusing_patterns])

        # Travel along the graph to find layers for fusing
        nodes = graph.get_topo_sorted_nodes()

        fusing_info: Dict[str, Tuple['BaseNode']] = {}
        fused_nodes = []  # nodes that are participating in fusing

        for node in nodes:
            # Skip if already in fusing
            if node in fused_nodes:
                continue
            # Start fusing search
            fusing_nodes = []  # nodes that are candidates for participating in fusing
            patterns = copy.deepcopy(self._fusing_patterns)
            next_nodes = [node]
            for i in range(max_layers_fusing):
                patterns = get_valid_fusing_patterns_for_node(patterns, next_nodes[0], i)
                if len(patterns) == 0:  # Give up if no more fusion pattern
                    break
                fusing_nodes.append(next_nodes[0])
                next_nodes = graph.get_next_nodes(fusing_nodes[-1])
                if len(next_nodes) != 1:  # Give up if node has more than one connection (not supported for fusion)
                    break

            # New fusion
            if is_valid_fusion(self._fusing_patterns, fusing_nodes):
                fused_op_id = FusingInfo.generate_fused_op_id(fusing_nodes)
                assert fused_op_id not in fusing_info, f"{fused_op_id} is already in fusing info: {fusing_info}"
                fusing_info[fused_op_id] = tuple(fusing_nodes)
                fused_nodes.extend(fusing_nodes)

        return FusingInfo(fusing_data=fusing_info, fusing_patterns=self._fusing_patterns)


def get_valid_fusing_patterns_for_node(fusing_patterns: List[List[Any]],
                                       node: 'BaseNode',
                                       idx: int = 0) -> List[List[Any]]:
    """
    Returns only the fusing patterns where a specific layer (at index idx) matches the given node — either by type or filter params.

    Args:
        fusing_patterns: supported fusing patterns
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


def is_valid_fusion(fusing_patterns: List[List[Any]], nodes: List['BaseNode']) -> bool:
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
