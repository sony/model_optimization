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
from model_compression_toolkit.core.common.graph.memory_graph.max_cut_astar import MaxCutAstar
from model_compression_toolkit.core.common.graph.memory_graph.memory_graph import MemoryGraph


def compute_graph_max_cut(memory_graph: MemoryGraph, n_iter: int = 50, astar_n_iter: int = 500, eps: float = 1e-2):
    max_cut_astar = MaxCutAstar(memory_graph=memory_graph)
    last_result = (0, None, None)  # TODO: consider creating an AstarResult class
    l_bound = memory_graph.memory_lbound_single_op
    u_bound = 2 * sum([t.total_size for t in memory_graph.b_nodes]) - l_bound
    it = 0
    while it < n_iter:
        estimate = (u_bound + l_bound) / 2
        max_cut_size, schedule, cuts = max_cut_astar.solve(estimate_factor=estimate, iter_limit=astar_n_iter)
        if schedule is None:
            return last_result

        next_u_bound = min(estimate, max_cut_size)
        last_result = (max_cut_size, schedule, cuts)

        if l_bound >= next_u_bound or l_bound * (1 + eps) > next_u_bound:
            return last_result

        it += 1

    return last_result
