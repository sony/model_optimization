# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
from torch.nn import Conv2d, ReLU, ReLU6, Linear, ConvTranspose2d, Sigmoid, Hardswish, Hardsigmoid, SiLU
from torch.nn.functional import sigmoid, relu, relu6, hardswish, hardsigmoid, silu
import torch
import operator
from typing import Tuple

from model_compression_toolkit import common
from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.graph.graph_matchers import NodeOperationMatcher, EdgeMatcher
from model_compression_toolkit.common.graph.base_node import BaseNode


class MarkActivation(common.BaseSubstitution):
    """
    There are cases we do not need to quantize a layer's output since there is an
    activation layer that follows it. Thus, in these cases we set the node's attribute
    that indicates whether the output should be quantized to False.
    """

    def __init__(self):
        """
        Matches: (DepthwiseConv2D,Conv2D,Conv2DTranspose,Dense)[activation=linear] -> (Activation,Relu)
                or
                Add -> (Activation,Relu)
        """
        source_node = (NodeOperationMatcher(Conv2d) |
                       NodeOperationMatcher(ConvTranspose2d) |
                       NodeOperationMatcher(Linear))

        activation_node = NodeOperationMatcher(ReLU) | \
                          NodeOperationMatcher(relu) | \
                          NodeOperationMatcher(ReLU6) | \
                          NodeOperationMatcher(relu6) | \
                          NodeOperationMatcher(SiLU) | \
                          NodeOperationMatcher(silu) | \
                          NodeOperationMatcher(Sigmoid) | \
                          NodeOperationMatcher(sigmoid) | \
                          NodeOperationMatcher(Hardsigmoid) | \
                          NodeOperationMatcher(hardsigmoid) | \
                          NodeOperationMatcher(Hardswish) | \
                          NodeOperationMatcher(hardswish)

        source_node_add = NodeOperationMatcher(torch.add) | \
                          NodeOperationMatcher(operator.add)

        super().__init__(
            matcher_instance=EdgeMatcher(source_node, activation_node) | EdgeMatcher(source_node_add, activation_node))

    def substitute(self,
                   graph: Graph,
                   edge: Tuple[BaseNode, BaseNode]) -> Graph:
        """
        Mark the first node in an edge that should not be quantized as so.
        This can be done due to the following reasons:
        1. The second node in the edge is an activation layer.

        Args:
            graph: Graph we apply the substitution on.
            edge: Edge where the first node's output should not be quantized.

        Returns:
            Graph after applying the substitution.
        """

        for nqc in edge[0].candidates_quantization_cfg:
            nqc.activation_quantization_cfg.enable_activation_quantization = False
        return graph
