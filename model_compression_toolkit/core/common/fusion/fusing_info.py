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
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OpQuantizationConfig
from model_compression_toolkit.constants import FUSED_LAYER_PATTERN, FUSED_OP_QUANT_CONFIG
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
    - 'fusing_patterns': The patterns to generate the fused operators from.
    - 'manual_fused_ops': List of sequence of node names to handle as fused ops (even if they are not part of the fusing patterns).
    - `fusing_data`: A dictionary mapping fused operation IDs to lists of nodes that belong to that operation.
    - `node_name_to_fused_op_id`: A dictionary mapping each node name to the ID of the fused operation it belongs to.

    """
    fusing_patterns: List[list[any]] = None
    manual_fused_ops: List[List[str]] = None
    fusing_data: Dict[str, Tuple['BaseNode']] = field(default_factory=dict)
    node_name_to_fused_op_id: Dict[str, str] = field(init=False, default_factory=dict)
    fused_op_id_to_quant_config: Dict[str, OpQuantizationConfig] = field(default_factory=dict)

    def __post_init__(self):
        """Validates and initializes mappings after dataclass instantiation."""
        self.fusing_patterns = self.fusing_patterns or []
        for op_id, op_nodes in self.fusing_data.items():
            assert isinstance(op_id, str) and op_id.startswith(FUSED_OP_ID_PREFIX), f"Found invalid fused op id: {op_id}"
            assert isinstance(op_nodes, tuple) and len(op_nodes) > 1, f"Found invalid fused op nodes: {op_nodes}"

        self._init_node_mapping()
        self._manual_fused_ops = self.manual_fused_ops or []
        del self.manual_fused_ops
        self._init_quantization_config_map()

    def _init_node_mapping(self) -> None:
        """
        Init the node-to-fused-node mapping based on the initial fusing data.
        """
        self.node_name_to_fused_op_id.clear()
        for op_id, nodes in self.fusing_data.items():
            for node in nodes:
                self.node_name_to_fused_op_id[node.name] = op_id

    def get_manual_nodes_to_fuse(self) -> List[List[str]]:
        """
        Get the list of node names to be fused manually.
        """
        return self._manual_fused_ops


    def add_manual_nodes_to_fuse(self, node_names: List[str]):
        """
        Add a list of node names to be fused manually.

        Args:
            node_names: List of nodes to be fused.

        """
        assert isinstance(node_names, list)
        assert all([isinstance(n, str) for n in node_names])
        assert node_names not in self._manual_fused_ops, f"{node_names} is already in manual fused ops: {self._manual_fused_ops}"
        self._manual_fused_ops.append(node_names)

    def _init_quantization_config_map(self) -> None:
        """
        Init the mapping between fused operation IDs and their quantization configurations.
        """
        self.fused_op_id_to_quant_config.clear()
        if self.fusing_patterns is not None:
            for op_id, nodes in self.fusing_data.items():
                self.set_fused_op_quantization_config(op_id, nodes)

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
            self.node_name_to_fused_op_id[node.name] = op_id

        # Update the quantization config mapping for this operation
        if self.fusing_patterns is not None:
            self.set_fused_op_quantization_config(op_id, nodes)

    def set_fused_op_quantization_config(self, op_id: str, nodes: Tuple['BaseNode']) -> None:
        """
        Set the quantization configuration for a given fused operation ID.

        Args:
            op_id (str): The identifier for the fused operation.
            nodes (Tuple[BaseNode]): The tuple of nodes that form the fused operation.
        """
        fusing_pattern = next((fp for fp in self.fusing_patterns if is_valid_fusion([fp.get(FUSED_LAYER_PATTERN)], nodes)), None)
        if fusing_pattern is not None:
            self.fused_op_id_to_quant_config[op_id] = fusing_pattern.get(FUSED_OP_QUANT_CONFIG)

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
        node_names = [n.name for n in nodes]
        if node_names in self._manual_fused_ops:
            self._manual_fused_ops.remove(node_names)

        for node in nodes:
            self.node_name_to_fused_op_id.pop(node.name, None)
        del self.fusing_data[op_id]
        self.fused_op_id_to_quant_config.pop(op_id, None)

    def get_fused_op_id_for_node(self, node_name: str) -> Optional[str]:
        """
        Get the name of the fused node containing the given original node name.

        Args:
            node_name: The name of a node from the original graph.

        Returns:
            The name of the fused node containing this node, or None if not fused.
        """
        return self.node_name_to_fused_op_id.get(node_name)

    def get_node_to_fused_node_map(self) -> Dict[str, str]:
        """
        Retrieve a copy of the mapping from original node names to fused node names.

        Returns:
            A dictionary mapping each original node name to its fused node name.
        """
        return self.node_name_to_fused_op_id.copy()

    def get_fusing_quantization_config_map(self) -> Dict[str, OpQuantizationConfig]:
        """
        Retrieve a copy of the mapping from fused operation IDs to their quantization configurations.

        Returns:
            A dictionary mapping each fused operation ID to its quantization configuration.
        """
        return self.fused_op_id_to_quant_config.copy()

    def get_fused_nodes(self, op_id: str) -> Optional[List['BaseNode']]:
        """
        Retrieve the list of nodes for a given fused operation ID.

        Args:
            op_id (str): The identifier for the fused operation.

        Returns:
            Optional[List[BaseNode]]: The list of nodes for the operation, or None if not found.
        """
        return self.fusing_data.get(op_id)

    def get_inner_fln_nodes(self) -> List['BaseNode']:
        """
        Returns a list of the nodes that are part but not the last node of an FLN.
        """
        # TODO: the order of the nodes is not gurenteed when returned as dict from get_all_fused_operations -
        #  then, removing the last one can cause issues
        return [node for nodes in self.get_all_fused_operations().values() for node in nodes[:-1]]

    def get_fused_op_quantization_config(self, op_id: str) -> OpQuantizationConfig:
        """
        Retrieve the quantization configuration for a given fused operation ID.

        Args:
            op_id (str): The identifier for the fused operation.

        Returns:
            OpQuantizationConfig: The quantization configuration for the operation, or None if not found.
        """
        return self.fused_op_id_to_quant_config.get(op_id)

    def is_node_in_fused_op(self, node: 'BaseNode') -> bool:
        """
        Check if a node is part of any fused operation.

        Args:
            node (BaseNode): The node to check.

        Returns:
            bool: True if the node is in any fused operation, False otherwise.
        """
        return any(node in nodes for nodes in self.fusing_data.values())

    def is_quantized_node_in_fln(self, node: 'BaseNode') -> bool:
        """
        Check whether a node inside an FLN and should be quantized.

        Args:
            node (BaseNode): The node to check.

        Returns:
            bool: True if the node is in any fused operation and should be quantized.
        """
        if self.is_node_in_fused_op(node):
            node_q_cfg = self.fused_op_id_to_quant_config[self.node_name_to_fused_op_id[node.name]]
            return node_q_cfg is not None and node_q_cfg.enable_activation_quantization

        return False

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
            valid_fusing_patterns = _get_fusing_layer_patterns(self.fusing_patterns)
            if not is_valid_fusion(valid_fusing_patterns, nodes, self._manual_fused_ops):
                raise ValueError(
                    f"Fused operation {op_id} does not match any valid fusing pattern "
                    f"from {valid_fusing_patterns}."
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
        valid_fusing_patterns = _get_fusing_layer_patterns(self.fusing_patterns)
        return is_valid_fusion(fusing_patterns=valid_fusing_patterns, nodes=nodes)

    def __repr__(self) -> str:
        """
        Return a string representation of the fusing information.
        """
        fusing_data_repr = "\n".join(
            f"  {op_id}: [{', '.join(node.name for node in nodes)}]"
            for op_id, nodes in self.fusing_data.items()
        )
        mapping_repr = ", ".join(
            f"{node} -> {op_id}" for node, op_id in self.node_name_to_fused_op_id.items()
        )
        return (
            f"FusingInfo(\n"
            f"  Total fused operations: {len(self.fusing_data)}\n"
            f"  Fusing Data:\n{fusing_data_repr}\n"
            f"  Node-to-Fused Mapping:\n  {mapping_repr}\n"
            f"  Manual fused ops:\n  {self._manual_fused_ops}\n"
            f")"
        )


class FusingInfoGenerator:
    def __init__(self, fusing_patterns: List[list] = None, manual_fused_ops: List[List[str]] = None):
        self._fusing_patterns = fusing_patterns or []
        assert isinstance(self._fusing_patterns, list)
        self._manual_fused_ops = manual_fused_ops or []
        assert isinstance(self._manual_fused_ops, list)

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
            return FusingInfo(fusing_patterns=self._fusing_patterns, manual_fused_ops=self._manual_fused_ops)

        # Extract fusing layer patterns
        fusing_layer_patterns = _get_fusing_layer_patterns(self._fusing_patterns)

        # Find max fusion
        max_layer_patterns = max([len(fusing_layer_pattern) for fusing_layer_pattern in fusing_layer_patterns])

        # Travel along the graph to find layers for fusing
        nodes = graph.get_topo_sorted_nodes()

        fusing_info: Dict[str, Tuple['BaseNode']] = {}
        fused_nodes = []  # nodes that are participating in fusing

        if len(self._fusing_patterns)>0:
            for node in nodes:
                # Skip if already in fusing
                if node in fused_nodes:
                    continue
                # Start fusing search
                fusing_nodes = []  # nodes that are candidates for participating in fusing
                patterns = copy.deepcopy(fusing_layer_patterns)
                next_nodes = [node]
                for i in range(max_layer_patterns):
                    patterns = get_valid_fusing_patterns_for_node(patterns, next_nodes[0], i)
                    if len(patterns) == 0:  # Give up if no more fusion pattern
                        break
                    fusing_nodes.append(next_nodes[0])
                    next_nodes = graph.get_next_nodes(fusing_nodes[-1])
                    if len(next_nodes) != 1:  # Give up if node has more than one connection (not supported for fusion)
                        break

                # New fusion
                if is_valid_fusion(fusing_layer_patterns, fusing_nodes):
                    fused_op_id = FusingInfo.generate_fused_op_id(fusing_nodes)
                    assert fused_op_id not in fusing_info, f"{fused_op_id} is already in fusing info: {fusing_info}"
                    fusing_info[fused_op_id] = tuple(fusing_nodes)
                    fused_nodes.extend(fusing_nodes)

        for manual_names in self._manual_fused_ops:
            manual_nodes = [graph.find_node_by_name(n) for n in manual_names]
            for n in manual_nodes:
                if len(n) != 1:
                    raise ValueError(f"Expected exactly one node, but got {len(n)}")
            manual_nodes = [n[0] for n in manual_nodes]

            # Remove any existing fused ops containing any of the manual nodes
            fused_ids_to_remove = {
                op_id for op_id, nodes in fusing_info.items()
                if any(node in nodes for node in manual_nodes)
            }
            for op_id in fused_ids_to_remove:
                del fusing_info[op_id]

            fused_op_id = FusingInfo.generate_fused_op_id(manual_nodes)
            assert fused_op_id not in fusing_info, f"{fused_op_id} is already in fusing info: {fusing_info}"
            fusing_info[fused_op_id] = tuple(manual_nodes)

        return FusingInfo(fusing_data=fusing_info,
                          fusing_patterns=self._fusing_patterns,
                          manual_fused_ops=self._manual_fused_ops)


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


def is_valid_fusion(fusing_patterns: List[List[Any]], nodes: List['BaseNode'], manual_fused_names: List[List[str]]=None) -> bool:
    """
    Check if the fusion is valid: exist in fusing_patterns
    Args:
        fusing_patterns: supported fusing patterns
        nodes: nodes which are participating in fusion
        manual_fused_names: list of nodes names to handle as a valid fusing op.
    Returns:
        whether the fusion in valid
    """
    node_names = [n.name for n in nodes]
    if any(manual == node_names for manual in (manual_fused_names or [])):
        return True

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


def _get_fusing_layer_patterns(fusing_patterns: List[Dict[Any, OpQuantizationConfig]]) -> List[List[Any]]:
    """
    Extracts the fusing layer patterns from the provided fusing patterns.
    Args:
        fusing_patterns: List of patterns of layers/LayerFilterParams to fuse and their mapping quantization config.

    Returns:
        supported fusing layer patterns
    """
    return [f.get(FUSED_LAYER_PATTERN) for f in fusing_patterns]
