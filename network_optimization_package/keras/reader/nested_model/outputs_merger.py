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

from typing import List

from network_optimization_package.common.graph.base_graph import Graph
from network_optimization_package.common.graph.node import Node
from network_optimization_package.keras.reader.connectivity_handler import OutTensor


def merge_models_outputs(inner_model_node: Node,
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
