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


import copy

from typing import List

from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.keras.reader.connectivity_handler import OutTensor


def merge_models_outputs(inner_model_node: BaseNode,
                         outer_graph: Graph,
                         inner_graph: Graph) -> List[OutTensor]:
    """
    Given two MultiDiGraphs (one of an outer model and the second of the inner model), merge their outputs into
    a single outputs list representing the output nodes that should be in a single graph after unrolling the
    inner graph.

    Args:
        outer_graph: MultiDiGraph of the outer model.
        inner_graph: MultiDiGraph of the inner model.
        inner_model_node: Node of the inner model in the graph of the outer model.

    Returns:
        Output nodes list that should be in a single graph after unrolling the inner graph.
    """

    res_model_outputs_list = copy.copy(outer_graph.get_outputs())
    # If the inner model is one the output nodes of the outer model, we need to update the final model outputs list
    # to contain the output nodes of the inner model as the output nodes of the final model. Also, we may need to update
    # the output tensor indices according to both the inner and outer outputs lists.
    for model_out_i, out_tensor in enumerate(outer_graph.get_outputs()):
        if inner_model_node == out_tensor.node:
            i = out_tensor.node_out_index
            new_out_tensor = inner_graph.get_outputs()[i]
            res_model_outputs_list[model_out_i] = new_out_tensor
    return res_model_outputs_list
