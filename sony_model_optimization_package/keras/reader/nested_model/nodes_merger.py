# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


import copy

# from sony_model_optimization_package.keras.reader.model_graph_info import ModelGraphInfo
from typing import List

from sony_model_optimization_package.common.graph.base_graph import Graph
from sony_model_optimization_package.common.graph.node import Node


def merge_models_nodes(inner_model_node: Node,
                       outer_graph: Graph,
                       inner_graph: Graph) -> List[Node]:
    """
    Given two MultiDiGraphs (one of an outer model and the second of the inner model), merge their nodes into
    a single nodes list representing the nodes that should be in a single MultiDiGraph after unrolling the inner graph.
    This is done by removing the inner model input nodes and the model node in the outer graph.
    Args:
        outer_graph: MultiDiGraph of the outer model.
        inner_graph: MultiDiGraph of the inner model.
        inner_model_node: Node of the inner model in the graph of the outer model.

    Returns:
        List of nodes that should be in a single graph after unrolling the inner graph.
    """
    # merge internal nodes
    res_nodes = copy.copy(list(outer_graph.nodes))
    res_nodes.extend(inner_graph.nodes)
    for input_node in inner_graph.get_inputs():  # inputs of inner model will no longer be needed.
        res_nodes.remove(input_node)
    # Remove the inner model node since we unroll it, and it's not going to be a node in the final graph.
    res_nodes.remove(inner_model_node)
    return res_nodes
