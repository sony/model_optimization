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


import copy
from typing import List, Dict, Callable

from networkx.algorithms.dag import topological_sort

import tensorflow as tf
from tensorflow.keras.layers import Layer, InputLayer
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.keras.constants import LAYER_NAME
from model_compression_toolkit.logger import Logger


class OperationHandler:
    """
    Class to handle conversions from graph nodes to Keras operators and retrieving them.
    """

    def __init__(self, graph: Graph, wrapper: Callable = None):
        """

        Args:
            graph: Graph to build its layers based on its nodes.
            wrapper: Wrapper to use for wrapping the layers.
        """

        # hold nodes after sorting them
        self.node_sort = list(topological_sort(graph))

        self.layer_to_node_dict = {}

        # hold dictionary from node to its equivalent Keras layer
        self.node_to_fw_op_dict = instance_builder(self.node_sort,
                                                   wrapper)

    def get_node_op_function(self, n: BaseNode) -> Layer:
        """
        Get the Keras layer that was built from the passed node.

        Args:
            n: Node to get its equivalent Keras layer.

        Returns:
            Keras layer for the node n.
        """

        if n.reuse:
            # If this node is a duplication of a reused layer, the reuse identifier needs to be removed from
            # the node's name as it was added to these nodes.
            original_name = '_'.join(n.name.split('_')[:-2])
            op_func = [v for k, v in self.node_to_fw_op_dict.items() if k.name == original_name][0]

        else:
            op_func = self.node_to_fw_op_dict[n]

        return op_func


def node_builder(n: common.BaseNode) -> Layer:
    """
    Build a Keras layer from a node.

    Args:
        n: Node to build its Keras layer

    Returns:
        Keras layer that was built from the node.
    """
    framework_attr = copy.copy(n.framework_attr)
    _layer_class = n.layer_class
    framework_attr[LAYER_NAME] = n.name  # Overwrite framework name to identical graph node name
    try:
        node_instance = _layer_class.from_config(framework_attr)  # Build layer from node's configuration.
    except Exception as e:
        print(e) # pragma: no cover
        Logger.error(
            f"Keras can not de-serialize layer {_layer_class} in order to build a static graph representation. This is probably because "
            f"your model contains custom layers which MCT doesn't support. Please provide a model without custom layers.") # pragma: no cover
    with tf.name_scope(n.name):
        # Add layer name to default weight name to avoid name duplications
        node_instance.build(n.input_shape)
    node_instance.set_weights(n.get_weights_list())
    node_instance.trainable = False  # Set all node as not trainable
    return node_instance


def instance_builder(toposort: List[BaseNode],
                     wrapper: Callable = None) -> Dict[BaseNode, Layer]:
    """
    Build a dictionary of nodes to their corresponding Keras
    layers, given a list of nodes.

    Args:
        toposort: List of nodes sorted topological to build their layers.
        wrapper: Wrapper to use for wrapping the layers.

    Returns:
        A dictionary of nodes to their corresponding Keras layers.
    """

    nodes_dict = dict()
    for n in toposort:
        if not n.reuse:  # Hold a single node in dictionary for all reused nodes from the same layer.
            keras_node = node_builder(n)
            if wrapper is not None:
                keras_node = wrapper(n, keras_node)
            nodes_dict.update({n: keras_node})

    return nodes_dict
