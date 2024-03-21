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

from typing import Dict

import copy
import numpy as np

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.pruning.pruning_section import PruningSectionMask
from model_compression_toolkit.logger import Logger


def build_pruned_graph(graph: Graph,
                       masks: Dict[BaseNode, np.ndarray],
                       fw_info: FrameworkInfo,
                       fw_impl: FrameworkImplementation) -> Graph:
    """
    Prunes the provided graph according to the given pruning output-channels masks.

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

    # Get the pruning sections.
    pruning_sections = graph_to_prune.get_pruning_sections(fw_impl=fw_impl)

    # Check that each entry node corresponds to a pruning section has an output-channel mask.
    if len(pruning_sections) != len(masks):
        Logger.critical(f"Expected to find the same number of masks as the number of pruning sections, but {len(masks)} masks were given for {len(pruning_sections)} pruning sections.") # progmra: no cover

    # Apply the pruning masks to each pruning section.
    for pruning_section in pruning_sections:

        # Retrieve the corresponding mask using the node's name (since we use a graph's copy).
        mask = [v for k, v in masks.items() if k.name == pruning_section.entry_node.name]
        if len(mask) != 1:
            Logger.critical(f"Expected to find a single node with name {pruning_section.entry_node.name} in masks dictionary, but found {len(mask)}.")
        mask = mask[0]

        # If the mask indicates that some channels are to be pruned, apply it.
        if np.any(mask == 0):
            section_mask = PruningSectionMask(entry_node_oc_mask=mask,
                                              exit_node_ic_mask=mask)
            pruning_section.apply_inner_section_mask(section_mask,
                                                     fw_impl,
                                                     fw_info)

    return graph_to_prune

