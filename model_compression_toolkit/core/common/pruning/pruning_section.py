from typing import List

import numpy as np

from model_compression_toolkit.core.common.graph.base_node import BaseNode


# class PruningSectionMask:
#     def __init__(self,
#                  input_node_ic_mask: np.ndarray,
#                  input_node_oc_mask: np.ndarray,
#                  output_node_oc_mask: np.ndarray
#                  ):
#         self.input_node_ic_mask = input_node_ic_mask
#         self.input_node_oc_mask = input_node_oc_mask
#         self.output_node_oc_mask = output_node_oc_mask
class PruningSectionMask:
    def __init__(self,
                 entry_input_mask: np.ndarray,
                 entry_output_mask: np.ndarray,
                 exit_input_mask: np.ndarray,
                 exit_output_mask: np.ndarray):

        self.entry_input_mask = entry_input_mask
        self.entry_output_mask = entry_output_mask
        self.exit_input_mask = exit_input_mask
        self.exit_output_mask = exit_output_mask

class PruningSection:

    def __init__(self,
                 entry_node:BaseNode,
                 intermediate_nodes: List[BaseNode],
                 exit_nodes: BaseNode):
        self.entry_node = entry_node
        self.intermediate_nodes = intermediate_nodes
        self.exit_node = exit_nodes

    def get_all_nodes(self):
        nodes = [self.entry_node]
        nodes.extend(self.intermediate_nodes)
        nodes.append(self.exit_node)
        return nodes

    def apply_inner_section_mask(self,
                                 pruning_section_mask: PruningSectionMask,
                                 fw_impl,
                                 fw_info):
        fw_impl.prune_entry_node(node=self.entry_node,
                                 output_mask=pruning_section_mask.entry_output_mask,
                                 fw_info=fw_info)

        for inter_node in self.intermediate_nodes:
            fw_impl.prune_intermediate_node(node=inter_node,
                                            input_mask=pruning_section_mask.entry_output_mask,
                                            output_mask=pruning_section_mask.entry_output_mask,
                                            fw_info=fw_info)

        fw_impl.prune_exit_node(self.exit_node,
                                input_mask=pruning_section_mask.exit_input_mask,
                                fw_info=fw_info)


