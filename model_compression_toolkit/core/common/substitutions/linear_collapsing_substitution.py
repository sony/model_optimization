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

from model_compression_toolkit.core import common


def linear_collapsing_substitute(graph: common.Graph,
                                 linear_collapsing_substitution: common.BaseSubstitution) -> common.Graph:
    """
    Apply a list of linear collapsing substitutions on a graph.
    We run on the graph and find matches. For each valid match we do substitution.
    After each substitution we find matches again on the transformed graph to look for new matches.
    This is because a node can participate in more than one match so after substitution
    the matches are not valid anymore, and we can find new matches.
    Args:
        graph: Graph to transform.
        linear_collapsing_substitution: substitution to apply on the graph.
    Returns:
        Transformed graph after applying all linear collapsing substitutions.
    """
    graph = copy.deepcopy(graph)
    matched_nodes = graph.filter(linear_collapsing_substitution.matcher_instance)
    matched_nodes_list = []
    match_indicator = True
    while len(matched_nodes) > 0 and match_indicator:
        match_indicator = False
        for matched_node in matched_nodes:
            if matched_node not in matched_nodes_list:
                # Substitute
                graph = linear_collapsing_substitution.substitute(graph, matched_node)
                matched_nodes_list.append(matched_node)
                match_indicator = True
                break
        # Find new matches on the transformed graph
        matched_nodes = graph.filter(linear_collapsing_substitution.matcher_instance)
    return graph