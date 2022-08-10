# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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

from typing import List

from model_compression_toolkit.core import common


def substitute(graph_to_substitute: common.Graph,
               substitutions_list: List[common.BaseSubstitution]) -> common.Graph:
    """
    Apply a list of substitutions on a graph.
    Args:
        graph: Graph to transform.
        substitutions_list: List of substitutions to apply on the graph.

    Returns:
        Transformed graph after applying all substitutions in substitutions_list.
    """

    graph = copy.deepcopy(graph_to_substitute)
    for substitution in substitutions_list:
        matched_nodes = graph.filter(substitution.matcher_instance)
        for idn in matched_nodes:
            graph = substitution.substitute(graph, idn)
    return graph