# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
from torch import nn
import torch.nn.functional as F

from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common import BaseNode, Graph, BaseSubstitution
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.pytorch.constants import *
from model_compression_toolkit.logger import Logger


class FunctionalLinear(BaseSubstitution):
    """
    Replace functional linear with Linear.
    """

    def __init__(self):
        """
        Matches: functional linear
        """
        func_node = NodeOperationMatcher(F.linear)
        super().__init__(matcher_instance=func_node)

    def substitute(self,
                   graph: Graph,
                   func_node: FunctionalNode) -> Graph:
        """
        Substitute functional.linear and its inputs with Linear.
        Args:
            graph: Graph we apply the substitution on.
            node: node that match the pattern in the substitution init.

        Returns:
            Graph after applying the substitution.
        """

        # Create new node of layer Linear
        if 1 not in func_node.weights:
            Logger.critical(f'Weight input missing for node {func_node.name}.')  # pragma: no cover
        # Extract index of kernel and bias according to tensor_input_allocs if they were input as kwargs. If
        # they were input as args, use their fixed positions.
        weight_index = func_node.tensor_input_allocs.index(KERNEL) if KERNEL in func_node.tensor_input_allocs else 1
        bias_index = func_node.tensor_input_allocs.index(BIAS) if BIAS in func_node.tensor_input_allocs else 2
        if weight_index not in func_node.weights:
            Logger.critical(f'Mismatch between tensor_input_allocs and weight index in node {func_node.name}.')  # pragma: no cover
        weight = func_node.weights[weight_index]
        bias = func_node.weights.get(bias_index)

        framework_attr = {
            IN_FEATURES: func_node.input_shape[0][-1],
            OUT_FEATURES: func_node.output_shape[0][-1],
            BIAS: bias is not None,
        }

        weights = {KERNEL: weight} if bias is None else {KERNEL: weight, BIAS: bias}

        new_node = BaseNode(
            name=func_node.name,
            framework_attr=framework_attr,
            input_shape=func_node.input_shape[0],
            output_shape=func_node.output_shape,
            weights=weights,
            layer_class=nn.Linear,
            has_activation=func_node.has_activation,
            reuse=func_node.reuse,
            reuse_group=func_node.reuse_group
        )

        graph.replace_node(func_node, new_node)
        return graph
