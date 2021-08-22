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


import tensorflow as tf
from tensorflow.python.keras.engine.functional import Functional
from tensorflow.python.keras.engine.node import Node as KerasNode
from tensorflow.python.keras.engine.sequential import Sequential

from network_optimization_package.common.graph.node import Node

keras = tf.keras
layers = keras.layers


def is_node_an_input_layer(node: Node) -> bool:
    """
    Checks if a node represents a Keras input layer.
    Args:
        node: Node to check if its an input layer.

    Returns:
        Whether the node represents an input layer or not.
    """
    if isinstance(node, Node):
        return node.layer_class == layers.InputLayer
    elif isinstance(node, KerasNode):
        return isinstance(node.layer, layers.InputLayer)
    else:
        raise Exception('Node to check has to be either a graph node or a keras node')


def is_node_a_model(node: Node) -> bool:
    """
    Checks if a node represents a Keras model.
    Args:
        node: Node to check if its a Keras model by itself.

    Returns:
        Whether the node represents a Keras model or not.
    """
    if isinstance(node, Node):
        return node.layer_class in [Functional, Sequential]
    elif isinstance(node, KerasNode):
        return isinstance(node.layer, Functional) or isinstance(node.layer, Sequential)
    else:
        raise Exception('Node to check has to be either a graph node or a keras node')

    # return node.layer_class in [Functional, Sequential]
