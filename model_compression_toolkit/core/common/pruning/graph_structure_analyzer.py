# # Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================
#
#
# from typing import List
# from networkx.algorithms.dag import topological_sort
#
# from model_compression_toolkit.core.common import Graph
# from model_compression_toolkit.core.common.graph.base_node import BaseNode
# from model_compression_toolkit.core.common.pruning.pruning_framework_implementation import \
#     PruningFrameworkImplementation
# from model_compression_toolkit.core.common.pruning.pruning_section import PruningSection
#
#
# class GraphStructureAnalyzer:
#     def __init__(self,
#                  graph: Graph,
#                  fw_impl: PruningFrameworkImplementation):
#
#         self.graph = graph
#         self.fw_impl = fw_impl
#
#     def get_pruning_sections(self) -> List[PruningSection]:
#         input_sections_nodes = self.get_pruning_sections_entry_nodes()
#         return [self._create_pruning_section(node) for node in input_sections_nodes]
#
#     def get_pruning_sections_entry_nodes(self):
#         return [n for n in topological_sort(self.graph) if self._is_node_prunable_entry(n)]
#
#     def _is_node_prunable_entry(self, node: BaseNode):
#         return self.fw_impl.is_node_entry_node(node) and self._is_node_topology_prunable(node)
#
#     def _is_node_topology_prunable(self,
#                                    node: BaseNode):
#         if not self.fw_impl.is_node_entry_node(node):
#             return False
#
#         next_node = node
#         while True:
#             out_edges = self.graph.out_edges(next_node)
#             # Check if the current node has only one outgoing edge
#             if len(out_edges) != 1:
#                 return False
#
#             next_node = out_edges[0].sink_node
#             # If the next node is prunable and has only one incoming edge, this topology is prunable.
#             if self.fw_impl.is_node_exit_node(next_node, node, self.graph.fw_info) and len(self.graph.in_edges(next_node)) == 1:
#                 return True
#
#             # If the next node is not an intermediate node or has more than one incoming edge, stop the check.
#             if not self.fw_impl.is_node_intermediate_pruning_section(next_node) or len(self.graph.in_edges(next_node)) != 1:
#                 return False
#         return False
#
#     def _create_pruning_section(self, start_node: BaseNode):
#         assert self.fw_impl.is_node_entry_node(start_node)
#         intermediate_nodes, next_node = self._find_intermediate_and_exit_nodes(start_node)
#         assert self.fw_impl.is_node_exit_node(next_node, start_node, self.graph.fw_info)
#         return PruningSection(entry_node=start_node,
#                               intermediate_nodes=intermediate_nodes,
#                               exit_node=next_node)
#
#     def _find_intermediate_and_exit_nodes(self, start_node: BaseNode):
#         intermediate_nodes = []
#         next_node = self.graph.out_edges(start_node)[0].sink_node
#         while not self.fw_impl.is_node_exit_node(next_node,
#                                                  start_node,
#                                                  self.graph.fw_info):
#             intermediate_nodes.append(next_node)
#             next_node = self.graph.out_edges(next_node)[0].sink_node
#
#         return intermediate_nodes, next_node
