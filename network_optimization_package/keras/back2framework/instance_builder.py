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
from typing import List, Dict

from networkx.algorithms.dag import topological_sort

from tensorflow.keras.layers import Layer
from network_optimization_package import common
from network_optimization_package.common import Graph, Node
from network_optimization_package.keras.constants import LAYER_NAME


class OperationHandler(object):
    """
    Class to handle conversions from graph nodes to Keras operators and retrieving them.
    """

    def __init__(self, graph: Graph):
        self.node_sort = list(topological_sort(graph))  # hold nodes after sorting them
        self.node_to_fw_op_dict = instance_builder(self.node_sort)  # hold dictionary from node to its equivalent
        # Keras layer

    def get_node_op_function(self, n: Node) -> Layer:
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


def node_builder(n: common.Node) -> Layer:
    """
    Build a Keras layer from a node.

    Args:
        n: Node to build its Keras layer

    Returns:
        Keras layer that was built from the node.
    """

    framework_attr = copy.copy(n.framework_attr)
    framework_attr[LAYER_NAME] = n.name  # Overwrite framework name to identical to graph node name
    node_instance = n.layer_class.from_config(framework_attr)  # Build layer from node's configuration.
    node_instance.build(n.input_shape)
    node_instance.set_weights(n.get_weights_list())
    node_instance.trainable = False  # Set all node as not trainable
    return node_instance


def instance_builder(toposort: List[Node]) -> Dict[Node, Layer]:
    """
    Build a dictionary of nodes to their corresponding Keras
    layers, given a list of nodes.

    Args:
        toposort: List of nodes sorted topological to build their layers.

    Returns:
        A dictionary of nodes to their corresponding Keras layers.
    """

    nodes_dict = dict()
    for n in toposort:
        if not n.reuse:  # Hold a single node in dictionary for all reused nodes from the same layer.
            nodes_dict.update({n: node_builder(n)})

    return nodes_dict
