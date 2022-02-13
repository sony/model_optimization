# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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


from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.graph.base_node import BaseNode
from model_compression_toolkit.keras.reader.nested_model.edges_merger import merge_models_edges
from model_compression_toolkit.keras.reader.nested_model.nodes_merger import merge_models_nodes
from model_compression_toolkit.keras.reader.nested_model.outputs_merger import merge_models_outputs


def merge_graphs(inner_model_node: BaseNode,
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

    return Graph(outer_graph.name,
                 nodes,
                 inputs,
                 outputs,
                 edges)
