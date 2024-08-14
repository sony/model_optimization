# Copyright 2024 Sony Semiconductors Israel, Inc. All rights reserved.
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

from typing import Tuple, Union
import numpy as np
import tensorflow as tf
from packaging import version
if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers.core import TFOpLambda
    from keras.src.layers import Multiply, Activation
else:
    from keras.layers.core import TFOpLambda
    from keras.layers import Multiply, Activation
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph, BaseNode, OutTensor
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher, \
    EdgeMatcher, NodeFrameworkAttrMatcher
from model_compression_toolkit.constants import REUSE, REUSE_GROUP
from model_compression_toolkit.core.keras.constants import FUNCTION, F_SWISH, ACTIVATION, SIGMOID


class MulSigmoidToSwish(common.BaseSubstitution):
    """
    Substitutes mul(x, sigmoid(x)) with swish.
    """

    def __init__(self):
        """
        Initializes the MulSigmoidToSwish substitution matcher instance.
        """
        mul_matcher = NodeOperationMatcher(tf.math.multiply) | NodeOperationMatcher(Multiply)
        activation_sigmoid = NodeOperationMatcher(Activation) & NodeFrameworkAttrMatcher(ACTIVATION, SIGMOID)
        sigmoid_matcher = NodeOperationMatcher(tf.sigmoid) | activation_sigmoid
        super().__init__(matcher_instance=EdgeMatcher(sigmoid_matcher, mul_matcher))

    def substitute(self,
                   graph: Graph,
                   sigmoid_mul_edge: Tuple[FunctionalNode, Union[FunctionalNode, BaseNode], int]) -> Graph:
        """
        Substitutes mul(x, sigmoid(x)) with swish.

        Args:
            graph: The graph on which the substitution is applied.
            sigmoid_mul_edge: edge between sigmoid and multiply nodes

        Returns:
            The modified graph after applying the substitution.
        """

        sigmoid_node, mul_node, _ = sigmoid_mul_edge
        if sigmoid_node in [o.node for o in graph.output_nodes]:
            # Sigmoid node in outputs -> Skip substitution.
            return graph

        input_node = graph.get_prev_nodes(sigmoid_node)[0]
        if len(graph.get_next_nodes(sigmoid_node)) > 1 or input_node not in graph.get_prev_nodes(mul_node):
            # Structure isn't mul(x, sigmoid(x)) -> Skip substitution.
            return graph
        _reuse_params = {REUSE: mul_node.reuse, REUSE_GROUP: mul_node.reuse_group}
        swish_node = FunctionalNode(f'swish__{sigmoid_node.name}_{mul_node.name}', {FUNCTION: F_SWISH},
                                    sigmoid_node.input_shape, mul_node.output_shape, {}, TFOpLambda,
                                    op_call_args=[], op_call_kwargs={}, functional_op=tf.nn.silu, **_reuse_params)

        graph.add_node(swish_node)

        # Replace functional conv node (and potentially add node) with Conv node.
        graph.reconnect_in_edges(sigmoid_node, swish_node)
        graph.reconnect_out_edges(mul_node, swish_node)
        graph.replace_output_node(current_node=mul_node, new_node=swish_node)
        graph.remove_edge(input_node, mul_node)
        graph.remove_edge(sigmoid_node, mul_node)
        graph.remove_node(sigmoid_node)
        graph.remove_node(mul_node)

        return graph

