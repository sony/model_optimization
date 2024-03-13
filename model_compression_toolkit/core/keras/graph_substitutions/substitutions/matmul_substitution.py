# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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

import numpy as np
import tensorflow as tf
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.keras.constants import TRANSPOSE_A, TRANSPOSE_B, \
    ADJOINT_A, ADJOINT_B, UNITS, USE_BIAS, KERNEL, ACTIVATION, LINEAR


class MatmulToDenseSubstitution(common.BaseSubstitution):
    """
    Replace a linear layer that has an activation function, with two nodes: same linear layer without
    an activation function, and a new activation layer to replace the function the linear node had.
    """

    def __init__(self):
        """
        Matches: tf.linalg.matmul
        """
        super().__init__(matcher_instance=NodeOperationMatcher(tf.linalg.matmul))

    def substitute(self,
                   graph: Graph,
                   matmul_node: FunctionalNode) -> Graph:
        """
        Replace tf.linalg.matmul with Tensor and const with Dense layer

        Args:
            graph: Graph we apply the substitution on.
            matmul_node: Node to replace.

        Returns:
            Graph after applying the substitution.
        """

        if len(graph.get_prev_nodes(matmul_node)) > 1:
            # matmul of 2 activation tensors -> can't replace with Dense layer
            return graph

        if matmul_node.framework_attr.get(ADJOINT_A, False) or matmul_node.framework_attr.get(ADJOINT_B, False):
            # MCT doesn't support complex tensors
            return graph

        if matmul_node.framework_attr.get(TRANSPOSE_A, False):
            # first input should be an activation tensor with batch axis, that shouldn't be transposed
            return graph

        # read const from matmul inputs
        w = matmul_node.weights.get(1)
        if w is None:
            Logger.critical(f"Matmul substitution failed: Unable to locate weight for node {matmul_node.name}.")  # pragma: no cover

        if len(w.shape) != 2:
            # weight tensor should be of shape (Cin, Cout)
            return graph

        # transpose const if "transpose_b" flag is True
        if matmul_node.op_call_kwargs.get(TRANSPOSE_B, False) or (
                len(matmul_node.op_call_args) >= 2 and matmul_node.op_call_args[1]):
            w = w.transpose()

        dense_node = BaseNode(matmul_node.name,
                              {UNITS: w.shape[1], USE_BIAS: False, ACTIVATION: LINEAR},
                              matmul_node.input_shape, matmul_node.output_shape,
                              {KERNEL: w}, tf.keras.layers.Dense,
                              reuse=matmul_node.reuse, reuse_group=matmul_node.reuse_group)

        graph.add_node(dense_node)
        graph.reconnect_in_edges(current_node=matmul_node,
                                 new_node=dense_node)
        graph.reconnect_out_edges(current_node=matmul_node,
                                  new_node=dense_node)
        graph.replace_output_node(current_node=matmul_node,
                                  new_node=dense_node)
        graph.remove_node(matmul_node)

        return graph


