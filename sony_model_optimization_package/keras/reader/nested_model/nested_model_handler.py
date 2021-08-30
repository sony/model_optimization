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


from sony_model_optimization_package.common.graph.base_graph import Graph
from sony_model_optimization_package.common.graph.node import Node
from sony_model_optimization_package.keras.reader.nested_model.edges_merger import merge_models_edges
from sony_model_optimization_package.keras.reader.nested_model.nodes_merger import merge_models_nodes
from sony_model_optimization_package.keras.reader.nested_model.outputs_merger import merge_models_outputs


def merge_graphs(inner_model_node: Node,
                 outer_graph: Graph,
                 inner_graph: Graph) -> Graph:
    """
    Given two MultiDiGraphs (one of an outer model and the second of the inner model), merge them into
    a single MultiDiGraph corresponding to the final model (namely, the outer model after unrolling the
    inner model, so it's no longer a nested model).

    Args:
        outer_graph: MultiDiGraph of the outer model.
        inner_graph: MultiDiGraph of the inner model.
        inner_model_node: Node of the inner model in the graph of the outer model.

    Returns:
        MultiDiGraph representing a model when the inner model is unrolled.
    """

    # merge nodes dictionaries into one nodes dictionary.
    nodes = merge_models_nodes(inner_model_node,
                               outer_graph,
                               inner_graph)

    # merge outputs dictionaries into one outputs dictionary.
    outputs = merge_models_outputs(inner_model_node,
                                   outer_graph,
                                   inner_graph)

    # Inner model can not be an input of the outer model. Thus, the final
    # inputs dictionary should be the outer model inputs dictionary.
    inputs = outer_graph.get_inputs()

    # merge edges lists into one edges list.
    edges = merge_models_edges(inner_model_node,
                               outer_graph,
                               inner_graph)

    return Graph(nodes,
                 inputs,
                 outputs,
                 edges)
