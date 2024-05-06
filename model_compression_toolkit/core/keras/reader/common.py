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


import tensorflow as tf
from packaging import version


if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.engine.input_layer import InputLayer
    from keras.src.engine.node import Node as KerasNode
    from keras.src.engine.functional import Functional
    from keras.src.engine.sequential import Sequential
else:
    from keras.engine.input_layer import InputLayer # pragma: no cover
    from keras.engine.node import Node as KerasNode # pragma: no cover
    from keras.engine.functional import Functional # pragma: no cover
    from keras.engine.sequential import Sequential # pragma: no cover

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.graph.base_node import BaseNode


def is_node_an_input_layer(node: BaseNode) -> bool:
    """
    Checks if a node represents a Keras input layer.
    Args:
        node: Node to check if its an input layer.

    Returns:
        Whether the node represents an input layer or not.
    """
    if isinstance(node, BaseNode):
        return node.is_match_type(InputLayer)
    elif isinstance(node, KerasNode):
        return isinstance(node.layer, InputLayer)
    else:
        Logger.critical('Node must be a graph node or a Keras node for input layer check.')  # pragma: no cover


def is_node_a_model(node: BaseNode) -> bool:
    """
    Checks if a node represents a Keras model.
    Args:
        node: Node to check if its a Keras model by itself.

    Returns:
        Whether the node represents a Keras model or not.
    """
    if isinstance(node, BaseNode):
        return node.is_match_type(Functional) or node.is_match_type(Sequential)
    elif isinstance(node, KerasNode):
        return isinstance(node.layer, Functional) or isinstance(node.layer, Sequential)
    else:
        Logger.critical('Node must be a graph node or a Keras node.')  # pragma: no cover

