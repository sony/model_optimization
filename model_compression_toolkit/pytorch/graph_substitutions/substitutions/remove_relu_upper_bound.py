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

from typing import List

from torch.nn import Conv2d, ReLU6, Linear, ConvTranspose2d, Hardtanh, ReLU
from torch.nn.functional import relu6, hardtanh, relu

from model_compression_toolkit import common
from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.graph.graph_matchers import NodeOperationMatcher, WalkMatcher
from model_compression_toolkit.common.graph.base_node import BaseNode
from model_compression_toolkit.common.constants import THRESHOLD
from model_compression_toolkit.pytorch.constants import KERNEL, BIAS, INPLACE, HARDTANH_MIN_VAL, HARDTANH_MAX_VAL

MATCHER = NodeOperationMatcher(ReLU6) | NodeOperationMatcher(relu6) | \
          NodeOperationMatcher(Hardtanh) | NodeOperationMatcher(hardtanh)


class RemoveReLUUpperBound(common.BaseSubstitution):
    """
    Remove ReLU upper bound if its activation threshold bounds it anyway at
    the same value.
    """

    def __init__(self):
        """
        Initialize a RemoveReLUUpperBound object.
        """
        super().__init__(matcher_instance=MATCHER)

    def substitute(self,
                   graph: Graph,
                   node: BaseNode) -> Graph:
        """
        Remove ReLU upper bound if its activation threshold bounds it anyway at
        the same value.

        Args:
            graph: Graph we apply the substitution on.
            node: Node to remove its bound.

        Returns:
            Graph after applying the substitution.
        """

        if node.type == ReLU6:
            node.layer_class = ReLU

        elif node.type == relu6:
            node.functional_op = relu

        if node.type == Hardtanh:
            node.layer_class = ReLU
            node.framework_attr = {'inplace': False}

        elif node.type == hardtanh:
            # TODO:
            # needs to be checked
            node.functional_op.__defaults__ = (0.0, None, False)
            node.functional_op = relu

        common.Logger.info(f'Removing upper bound of {node.name}')
        return graph
