# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
import copy
from typing import List, Tuple, Dict, Set
from time import time

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.constants import DUMMY_TENSOR, DUMMY_NODE
from model_compression_toolkit.core.common.graph.memory_graph.cut import Cut
from model_compression_toolkit.core.common.graph.memory_graph.memory_element import MemoryElements
from model_compression_toolkit.core.common.graph.memory_graph.memory_graph import ActivationMemoryTensor, MemoryGraph


class DummyType:
    """
    A dummy type class to use for dummy nodes layer type.
    """
    pass


class DummyBaseNodeGenerator:
    """
    A Dummy node which represents a "Side A" node (operation) in the memory graph for max cut AStar algorithm.
    """
    def __init__(self):
        self.counter = 0

    def __iter__(self) -> BaseNode:
        """
        A generator of sequentially named dummy nodes.
        """
        while True:
            yield BaseNode(name=f"{DUMMY_NODE}_{self.counter}",
                           framework_attr={},
                           input_shape=tuple(),
                           output_shape=tuple(),
                           weights={},
                           layer_class=DummyType,
                           has_activation=False)

            self.counter += 1


class DummyActivationMemoryTensorGenerator:
    """
     A Dummy node which represents a "Side B" node (activation tensor) in the memory graph for max cut AStar algorithm.
     """
    def __init__(self):
        self.counter = 0

    def __iter__(self) -> ActivationMemoryTensor:
        """
        A generator of sequentially named dummy nodes.
        """
        while True:
            yield ActivationMemoryTensor(shape=tuple(),
                                         node_name=f"{DUMMY_TENSOR}_{self.counter}",
                                         node_output_index=0,
                                         init_size_to_zero=True)

            self.counter += 1


class MaxCutAstar:
    """
    Implements the AStar solver and all the relevant utility methods to run a search for schedule and max cut
    on a model memory graph.
    The solver returns a schedule, the max cut of the schedule and a set of all the cuts that are developed during
    the computation of the model according to the returned schedule.
    """

    def __init__(self, memory_graph: MemoryGraph):
        """
        Args:
            memory_graph: A MemoryGraph object to run the search on.
        """

        self.memory_graph = memory_graph

        # In order to run Astar search on the graph we need to define a single source and a single target.
        # Dummy nodes are used for this purpose.
        # We do it a bit different from the original scheduler implementation.
        gen_a = iter(DummyBaseNodeGenerator())
        gen_b = iter(DummyActivationMemoryTensorGenerator())

        # Source Cut
        src_dummy_a = next(gen_a)
        src_dummy_b = next(gen_b)
        edges_src_ab = [(src_dummy_a, src_dummy_b)]
        edges_src_ba = [(src_dummy_b, src_a) for src_a in memory_graph.sources_a]

        # Target Cut (Adding 2 consecutive dummy nodes so the final cut will include only dummy tensors).
        target_dummy_a = next(gen_a)
        target_dummy_a2 = next(gen_a)
        target_dummy_b = next(gen_b)
        target_dummy_b2 = next(gen_b)
        edges_target_fix_ba = [(t, target_dummy_a) for t in memory_graph.sinks_b]
        edges_target_ab = [(target_dummy_a, target_dummy_b), (target_dummy_a2, target_dummy_b2)]
        edges_target_ba = [(target_dummy_b, target_dummy_a2)]

        # Update memory graph
        # New nodes
        self.memory_graph.add_nodes_to_a([src_dummy_a, target_dummy_a, target_dummy_a2])
        self.memory_graph.add_nodes_to_b([src_dummy_b, target_dummy_b, target_dummy_b2])
        # New Edges
        self.memory_graph.add_edges(edges_src_ab + edges_src_ba + edges_target_fix_ba + edges_target_ab + edges_target_ba)
        self.memory_graph.update_sources_a()
        self.memory_graph.update_sinks_b()

        self.src_cut = Cut([src_dummy_a], {src_dummy_a}, MemoryElements(elements={src_dummy_b}, total_size=0))
        self.target_cut = Cut([], set(), MemoryElements(elements={target_dummy_b, target_dummy_b2},
                                                        total_size=0))

    def solve(self, estimate: float, iter_limit: int = 500, time_limit: int = None) -> Tuple[List[BaseNode], float, List[Cut]]:
        """
        The AStar solver function. This method runs an AStar-like search on the memory graph,
        using the given estimate as a heuristic gap for solutions to consider.

        Args:
            estimate: Cut size estimation to consider larger size of nodes in each
                expansion step, in order to fasten the algorithm divergence towards a solution.
            iter_limit: An upper limit for the number of expansion steps that the algorithm preforms.
            time_limit: Optional time limit to the solver. Defaults to None which means no limit.

        Returns: A solution (if found within the steps limit) which contains:
        - A schedule for computation of the model (List of nodes).
        - The cost of a max cut of the found schedule.
        - All the cuts that are developed during the computation on the model according to the found schedule (List of Cuts).

        """

        open_list = {self.src_cut}
        closed_list = set()
        costs = {self.src_cut: self.src_cut.memory_size()}
        routes = {self.src_cut: [self.src_cut]}

        expansion_count = 0

        t1 = time()
        while expansion_count < iter_limit and len(open_list) > 0:
            if time_limit is not None and time() - t1 > time_limit:
                # TODO: add test for this.
                raise TimeoutError  # pragma: no cover
            # Choose next node to expand
            next_cut = self._get_cut_to_expand(open_list, costs, routes, estimate)

            cut_cost = costs[next_cut]
            cut_route = routes[next_cut]

            if next_cut == self.target_cut:
                return self._remove_dummy_nodes_from_path(cut_route[0].op_order), cut_cost,\
                       list(set([self._remove_dummy_tensors_from_cut(c) for c in cut_route]))

            if self.is_pivot(next_cut):
                # Can clear all search history
                open_list.clear()
                closed_list.clear()
                routes = {}
            else:
                # Can remove only next_cut and put it in closed_list
                open_list.remove(next_cut)
                del routes[next_cut]
                closed_list.add(next_cut)

            # Expand the chosen cut
            expanded_cuts = self.expand(next_cut)
            expansion_count += 1

            # Only consider nodes that where not already visited
            for c in filter(lambda _c: _c not in closed_list, expanded_cuts):
                cost = self.accumulate(cut_cost, c.memory_size())
                if c not in open_list:
                    self._update_expanded_node(c, cost, cut_route, open_list, costs, routes)
                # TODO maxcut: this isn't covered in the coverage test. check if needed and remove no cover
                elif self.ordering(cost, costs[c]):  # pragma: no cover
                    # If we already saw this cut during the search with a larger cost, then we want to update the order
                    # of the schedule in the cut
                    # Remove call - removes the cut with the same memory elements but different ordering from open
                    # Then - adds the cut with the improved ordering from open
                    open_list.remove(c)
                    self._update_expanded_node(c, cost, cut_route, open_list, costs, routes)

        # Halt or No Solution
        # TODO maxcut: this isn't covered in the coverage test. Add test and remove no cover
        return None, 0, None  # pragma: no cover

    @staticmethod
    def _update_expanded_node(cut: Cut, cost: float, route: List[Cut], open_list: Set[Cut],
                              costs: Dict[Cut, float], routes: Dict[Cut, List[Cut]]):
        """
        An auxiliary method for updating search data structures according to an expanded node.

        Args:
            cut: A cut to expand the search to.
            cost: The cost of the cut.
            route: The rout to the cut.
            open_list: The search open set.
            costs: The search utility mapping between cuts and their cost.
            routes: The search utility mapping between cuts and their routes.

        """
        open_list.add(cut)
        costs.update({cut: cost})
        routes.update({cut: [cut] + route})

    def _get_cut_to_expand(self, open_list: Set[Cut], costs: Dict[Cut, float], routes: Dict[Cut, List[Cut]],
                           estimate: float) -> Cut:
        """
        An auxiliary method for finding a cut for expanding the search out of a set of potential cuts for expansion.

        Args:
            open_list: The search open list.
            costs: The search utility mapping between cuts and their cost.
            routes: The search utility mapping between cuts and their routes.
            estimate: Cut size estimation to set extended boundaries on the potential cuts to expand.

        Returns: A sorted list of potential cuts for expansion (ordered by lowest cost first).

        """
        max_cut_len = max([len(routes[c]) for c in open_list])
        ordered_cuts_list = sorted(open_list,
                                   key=lambda c: (self.accumulate(costs[c], self.estimate(c, estimate)),
                                                  max_cut_len - len(routes[c]),
                                                  c.sorted_elements_signature))

        assert len(ordered_cuts_list) > 0
        return ordered_cuts_list[0]

    def clean_memory_for_next_step(self, cut: Cut) -> Cut:
        """
         An auxiliary function that removes irrelevant memory elements from a cut.
         The memory elements are irrelevant if all operations depended on them have already been executed.

        Args:
            cut: A Cut to remove elements from.

        Returns: A new Cut with updated memory elements list.

        """

        cut_records_names = cut.get_record_names()
        filtered_memory_elements = set(filter(lambda elm: not all(child.name in cut_records_names for child in
                                                                  self.memory_graph.activation_tensor_children(elm)),
                                              cut.mem_elements.elements))

        return Cut(cut.op_order, cut.op_record,
                   mem_elements=MemoryElements(filtered_memory_elements,
                                               sum([elm.total_size for elm in filtered_memory_elements])))

    def can_expand(self, op_node: BaseNode, cut: Cut) -> bool:
        """
        Checks whether a cut can be expanded by adding an operation node to it, that is,
        all the required memory elements for the operation computation exist in the cut.

        Args:
            op_node: An operation node to check if it can expand the cut.
            cut: A cut that contains the op_node.

        Returns: Whether the cut can be expanded by expanding the op_node.
        """

        clean_cut = self.clean_memory_for_next_step(cut)
        return op_node not in cut.op_record and len(cut.mem_elements.elements) > 0 and \
               all([parent_mem_element in clean_cut.mem_elements.elements for parent_mem_element in self.memory_graph.operation_node_parents(op_node)])

    def expand(self, cut: Cut) -> List[Cut]:
        """
        Expends the search with the given cut.

        Args:
            cut: A cut to expand the search to.

        Returns: A list of successors of the expanded cut.

        """
        clean_cut = self.clean_memory_for_next_step(cut)

        # candidates for expansion are children of the memory elements from the cleaned cut that can be expanded
        candidates = []
        for mem_element in clean_cut.mem_elements.elements:
            child_ops = self.memory_graph.activation_tensor_children(mem_element)
            ops_to_expand = list(filter(lambda op: self.can_expand(op, clean_cut) and op not in candidates, child_ops))
            candidates.extend(ops_to_expand)

        # for each candidate a cut is returned with the candidate expanded
        # (operation is added to record / order and resulting memory elements added to memory elements)
        next_cuts = []
        for candidate in candidates:
            op_order = clean_cut.op_order + [candidate]

            op_record = clean_cut.op_record.copy()
            op_record.add(candidate)

            mem_elements = copy.copy(clean_cut.mem_elements)
            mem_elements.add_elements_set(set(self.memory_graph.operation_node_children(candidate)))

            expanded_cut = Cut(op_order, op_record, mem_elements)
            next_cuts.append(expanded_cut)

        return next_cuts

    def is_pivot(self, cut: Cut) -> bool:
        """
        returns true if Cut is a pivot i.e. the cut must be in the selected route.
        If all memory elements in the cut have the same parent it is a pivot.

        Args:
            cut: A Cut to check whether it is a pivot.

        Returns: True if the given cut is a pivot.

        """

        clean_cut = self.clean_memory_for_next_step(cut)

        mem_elements_parents = list(map(lambda mem_elm: self.memory_graph.activation_tensor_parents(mem_elm), clean_cut.mem_elements.elements))
        unique_parents = set(sum(mem_elements_parents, []))

        return len(unique_parents) == 1

    @staticmethod
    def accumulate(cost_1: float, cost_2: float) -> float:
        """
        A function that defines the accumulation method of costs for the Astar search.
        We take the maximum memory requirement of the schedule, in order to find the minimal maximum out of all possible schedules.

        Args:
            cost_1: The first schedule option's cost.
            cost_2: The second schedule option's cost.

        Returns: The maximum out of the two costs.

        """
        return max(cost_1, cost_2)

    @staticmethod
    def ordering(cost_1, cost_2) -> bool:
        """
        A function that defines the ordering method of costs for the Astar search.
        We consider a lowest-first order.

        Args:
            cost_1: The first schedule option's cost.
            cost_2: The second schedule option's cost.

        Returns: True if the first cost is smaller than the second one, else otherwise.

        """
        # TODO maxcut: this isn't covered in the coverage test. check if needed and remove no cover
        return cost_1 < cost_2  # pragma: no cover

    @staticmethod
    def estimate(cut: Cut, estimate: float) -> float:
        """
        A function that defines the estimation gap for the Astar search.
        The estimation gap is used to sort the cuts that are considered for expanding the search in each iteration.

        Args:
            cut: A cut (not used in the default implementation, but can be used if overriding the method to consider
                the actual cut in the estimation computation).
            estimate: The given estimate to the search.

        Returns: An estimation value.

        """
        return estimate

    @staticmethod
    def get_init_estimate(memory_graph: MemoryGraph) -> float:  # pragma: no cover
        """
        Returns an initial estimation value, which is based on the memory graph's upper and lower bounds.

        Args:
            memory_graph: A MemoryGraph object.

        Returns: An initial estimate value.

        """
        # TODO maxcut: this isn't covered in the coverage test. check if needed and remove no cover
        l_bound = memory_graph.memory_lbound_single_op
        u_bound = 2 * sum([t.total_size for t in memory_graph.b_nodes]) - l_bound
        return (u_bound + l_bound) / 2

    @staticmethod
    def _remove_dummy_nodes_from_path(path: List[BaseNode]) -> List[BaseNode]:
        """
        An auxiliary method which removes dummy nodes from a given list of nodes (a path in the graph).

        Args:
            path: A path in the graph (list of nodes).

        Returns: The same list without any dummy nodes.

        """
        return list(filter(lambda n: DUMMY_NODE not in n.name, path))

    @staticmethod
    def _remove_dummy_tensors_from_cut(cut: Cut) -> Cut:
        """
        An auxiliary method which removes dummy nodes from a given cut.

        Args:
            cut: A cut to remove dummy nodes from.

        Returns: The same cut without dummy nodes elements.

        """
        filtered_memory_elements = set([elm for elm in cut.mem_elements.elements if DUMMY_TENSOR not in elm.node_name])

        return Cut(cut.op_order, cut.op_record,
                   mem_elements=MemoryElements(filtered_memory_elements,
                                               sum([elm.total_size for elm in filtered_memory_elements])))


