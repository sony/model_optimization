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

# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.
if tf.__version__ < "2.6":
    from tensorflow.python.keras.engine.node import Node as KerasNode
    from tensorflow.keras.layers import InputLayer
    from tensorflow.python.keras.engine.functional import Functional
    from tensorflow.python.keras.engine.sequential import Sequential
else:
    from keras.engine.input_layer import InputLayer
    from keras.engine.node import Node as KerasNode
    from keras.engine.functional import Functional
    from keras.engine.sequential import Sequential

from model_compression_toolkit.common.graph.base_node import BaseNode



def is_node_an_input_layer(node: BaseNode) -> bool:
    """
    Checks if a node represents a Keras input layer.
    Args:
        node: Node to check if its an input layer.

    Returns:
        Whether the node represents an input layer or not.
    """
    if isinstance(node, BaseNode):
        return node.type == InputLayer
    elif isinstance(node, KerasNode):
        return isinstance(node.layer, InputLayer)
    else:
        raise Exception('Node to check has to be either a graph node or a keras node')


def is_node_a_model(node: BaseNode) -> bool:
    """
    Checks if a node represents a Keras model.
    Args:
        node: Node to check if its a Keras model by itself.

    Returns:
        Whether the node represents a Keras model or not.
    """
    if isinstance(node, BaseNode):
        return node.type in [Functional, Sequential]
    elif isinstance(node, KerasNode):
        return isinstance(node.layer, Functional) or isinstance(node.layer, Sequential)
    else:
        raise Exception('Node to check has to be either a graph node or a keras node')

