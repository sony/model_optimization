# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Any

import tensorflow as tf

if tf.__version__ < "2.6":
    from tensorflow.python.keras.layers.core import TFOpLambda, SlicingOpLambda
    from tensorflow.python.keras.engine.keras_tensor import KerasTensor
    from tensorflow.python.keras.engine.node import Node as KerasNode
else:
    from keras.layers.core import TFOpLambda, SlicingOpLambda
    from keras.engine.keras_tensor import KerasTensor
    from keras.engine.node import Node as KerasNode

from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode

keras = tf.keras
layers = keras.layers

REUSED_IDENTIFIER = '_reused_'


def build_node(node: KerasNode,
               node_name_to_node: dict) -> BaseNode:
    """
    Build a node from a Keras node. A node contains all information to reconstruct the layer it's representing
    in a model:
    operation, layer configuration, path for instantiating the Keras layer the node has, weights, group of other
    nodes if it's a reused layer,
    input/output shape.
    Args:
        node: Node in the graph of a Keras model.
        node_name_to_node: Dictionary of already created nodes aims to identify reused layers.

    Returns:
        Graph node that was built from the Keras node.
    """
    keras_layer = node.layer  # get the layer the node represents.
    layer_config = keras_layer.get_config()  # layer configuration to reconstruct it.
    op_call_args = node.call_args
    op_call_kwargs = node.call_kwargs
    layer_class = type(keras_layer)  # class path to instantiating it in back2framework.
    weights = {v.name: v.numpy() for v in keras_layer.weights}  # layer's weights

    # If it's a node representing a reused layer, several nodes will contain the same layer instance.
    # Thus, the first time a node of a reused layer is being created, it's being build as a node of a non-reused layer,
    # while other nodes of this layer will be created as nodes of a reused layer with the suffix "_reused_i"
    # where i is the input/output index of the layer.
    is_reused = keras_layer.name in node_name_to_node
    if is_reused:
        # Mark the "base" node with its reused group.
        node_name_to_node[keras_layer.name].reuse_group = keras_layer.name
        io_index = 1
        while keras_layer.name + REUSED_IDENTIFIER + str(io_index) in node_name_to_node:  # find next unused io index
            io_index = io_index + 1
        reuse_group = keras_layer.name  # by the layer name we can gather nodes of this reused layer
        node_name = keras_layer.name + REUSED_IDENTIFIER + str(io_index)
    else:
        io_index = 0  # for non reused layers input/output index is 0
        reuse_group = None
        node_name = keras_layer.name
    input_shape = keras_layer.get_input_shape_at(io_index)
    output_shape = keras_layer.get_output_shape_at(io_index)

    if layer_class in [TFOpLambda, SlicingOpLambda]:
        # Some functional ops (such as tf.concat) should receive the input tensors as a list
        # and some are not (such as tf.multiply), so each FunctionalNode holds
        # a flag to indicate that.
        inputs_as_list = __is_functional_inputs_a_list(op_call_args)
        # Do not hold the tensors that are in op_call_args as they are
        # not needed. Thus, if the first element in op_call_args is a list of
        # Keras tensors, remove it from op_call_args.
        op_call_args = op_call_args[int(inputs_as_list):]
        node = FunctionalNode(node_name,
                              layer_config,
                              input_shape,
                              output_shape,
                              weights,
                              layer_class,
                              [arg for arg in op_call_args if not isinstance(arg, KerasTensor)],  # Do not hold the tensors that are in op_call_args
                              {k: v for k, v in op_call_kwargs.items() if not isinstance(v, KerasTensor)}, # In TF2.5 tensors are in kwargs as well.
                              is_reused,
                              reuse_group,
                              functional_op=keras_layer.function,
                              inputs_as_list=inputs_as_list)
    else:
        node = BaseNode(node_name,
                        layer_config,
                        input_shape,
                        output_shape,
                        weights,
                        layer_class,
                        is_reused,
                        reuse_group)

    node_name_to_node[node_name] = node
    return node


def __is_functional_inputs_a_list(op_call_args: Any) -> bool:
    """
    Check whether the input tensors should be passed as a list
    or not.

    Args:
        op_call_args: Arguments list to check.

    Returns:
        Whether the input tensors should be passed as a list or not.
    """

    if len(op_call_args) > 0 and isinstance(op_call_args[0], list):
        inputs_as_list = True
        for arg in op_call_args[0]:
            inputs_as_list = inputs_as_list and isinstance(arg, KerasTensor)
        return inputs_as_list
    return False
