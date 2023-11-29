from typing import Dict

import copy
import numpy as np

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.pruning.pruning_section import PruningSectionMask


def build_pruned_graph(graph: Graph,
                       masks: Dict[BaseNode, np.ndarray],
                       fw_info: FrameworkInfo,
                       fw_impl: FrameworkImplementation) -> Graph:
    """
    Prunes the provided graph according to the given pruning masks.

    Args:
        graph: The original computational graph to be pruned.
        masks: A dictionary mapping each prunable node to its pruning mask.
        fw_info: Framework-specific information object.
        fw_impl: Framework-specific implementation object.

    Returns:
        A pruned copy of the original computational graph.
    """

    # Create a deep copy of the graph to avoid modifying the original graph.
    graph_to_prune = copy.deepcopy(graph)

    # Get the prunable nodes and the corresponding pruning sections.
    prunable_nodes = graph_to_prune.get_pruning_sections_entry_nodes(fw_info=fw_info, fw_impl=fw_impl)
    pruning_sections = graph_to_prune.get_pruning_sections(fw_info=fw_info, fw_impl=fw_impl)

    # Check that each prunable node corresponds to a pruning section.
    assert len(pruning_sections) == len(prunable_nodes)

    # Apply the pruning masks to each pruning section.
    for input_node, pruning_section in zip(prunable_nodes, pruning_sections):

        # Retrieve the corresponding mask using the node's name (since we use a graph's copy).
        mask = [v for k, v in masks.items() if k.name == input_node.name]
        assert len(mask) == 1, f"Expected to find a single node with name {input_node.name} in masks dictionary but found {len(mask)}"
        mask = mask[0]

        # If the mask indicates that some channels are to be pruned, apply it.
        if np.any(mask == 0):
            section_mask = PruningSectionMask(entry_input_mask=None,
                                              entry_output_mask=mask,
                                              exit_input_mask=mask,
                                              exit_output_mask=None)
            pruning_section.apply_inner_section_mask(section_mask, fw_impl, fw_info)

    # Return the pruned graph.
    return graph_to_prune

