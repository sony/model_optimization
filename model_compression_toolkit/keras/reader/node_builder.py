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


import tensorflow as tf
from tensorflow.python.keras.engine.node import Node as KerasNode

from model_compression_toolkit.common.graph.node import Node

keras = tf.keras
layers = keras.layers

REUSED_IDENTIFIER = '_reused_'


def build_node(node: KerasNode,
               node_name_to_node: dict) -> Node:
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
    op_call_args = node.call_kwargs
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

    node = Node(node_name,
                layer_config,
                input_shape,
                output_shape,
                weights,
                layer_class,
                is_reused,
                reuse_group,
                op_call_args)

    node_name_to_node[node_name] = node

    return node
