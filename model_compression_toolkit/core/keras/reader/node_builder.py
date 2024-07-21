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
from copy import copy

from typing import Any, List, Dict, Union, Tuple

import tensorflow as tf
from tensorflow.python.util import tf_inspect

import numpy as np
from packaging import version

from model_compression_toolkit.core.keras.custom_layer_validation import is_keras_custom_layer
from model_compression_toolkit.core.keras.tf_tensor_numpy import tf_tensor_to_numpy as to_numpy
from model_compression_toolkit.logger import Logger

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers.core import TFOpLambda, SlicingOpLambda
    from keras.src.engine.keras_tensor import KerasTensor
    from keras.src.engine.node import Node as KerasNode
else:
    from keras.layers.core import TFOpLambda, SlicingOpLambda
    from keras.engine.keras_tensor import KerasTensor
    from keras.engine.node import Node as KerasNode

from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode

keras = tf.keras
layers = keras.layers

REUSED_IDENTIFIER = '_reused_'

is_const = lambda x: isinstance(x, (tf.Variable, tf.Tensor, np.ndarray, tuple, list))
is_tensor = lambda x: isinstance(x, KerasTensor)


def get_tf_function_symbols() -> List[str]:
    """
    Create a list of tf function symbols, as they are created in the TFOpLambda layer. The
    symbols are serializations of the function names.

    Returns:
         A list of TF function symbols,
    """
    return [TFOpLambda(f).symbol for f in [tf.add, tf.multiply, tf.subtract, tf.divide,
                                           tf.truediv, tf.pow, tf.matmul]]


def get_kwargs2index(tfoplambda_layer: TFOpLambda) -> Dict[str, int]:
    """
    Positional weights are saved according to their index in the node's call arguments, so
    need to know the function arguments' names in case the weights are in the kwargs.

    Args:
        tfoplambda_layer: TFOpLambda layer.

    Returns:
        A dictionary with argument number and index: {arg_name: arg_index}.
    """

    full_args = tf_inspect.getfullargspec(tfoplambda_layer.function).args

    return {arg_name: i for i, arg_name in enumerate(full_args)}


def _extract_const_attrs_from_kwargs(op_call_kwargs: Dict[str, Any],
                                     kwarg2index: Dict[str, int],
                                     weights: Dict[Union[str, int], Any]) -> Dict[str, Any]:
    """
    Extract const weights of the layer from the operator's key arguments dictionary.
    This function extracts the attributes, updates the nodes weights dictionary and removes them from the original
    kwargs mapping.

    Args:
        op_call_kwargs: A mapping of the operator key arguments.
        kwarg2index: A dictionary with argument number and index: {arg_name: arg_index}.
        weights: Node weights mapping. This dictionary is modified by this function.

    Returns: A modified operator key arguments mapping.

    """

    # read weights from call kwargs
    for k, v in op_call_kwargs.items():
        if is_const(v):
            # if k in kwarg2index:
            weights.update({kwarg2index[k]: to_numpy(v, is_single_tensor=True)})

    # remove weights and KerasTensors from op_call_kwargs
    op_call_kwargs = {k: v for k, v in op_call_kwargs.items()
                      if not (kwarg2index.get(k) in weights or is_tensor(v))}

    return op_call_kwargs


def _build_arguments_alloc(n: KerasNode, inputs_as_list: bool, kwarg2index: Dict[str, int]) -> List:
    """
    Builds arguments allocation list.
    In Keras, if there is any argument that is a constant, we convert all arguments and inputs to be
    considered as op kwargs for simpler reconstruction of the model from the graph later.
    Therefore, we build a location list that includes the argument names (keys).
    If the input is a list, then we don't need to save the keys, since we can assume that all possible constant
    arguments are within the first argument (the list) and are stored by their position in the list.

    Args:
        n: fx node.
        inputs_as_list: Is node's inputs are a list.

    Returns:
        A list of argument allocations in the node's inputs.

    """

    tensor_input_alloc = []
    op_call_args = list(n.call_args)
    if not inputs_as_list:
        sorted_kwargs_pos = sorted(kwarg2index.items(), key=lambda x: x[1])
        tensor_input_alloc = [k for k, _ in sorted_kwargs_pos[:len(op_call_args)]]
        for k, idx in sorted_kwargs_pos[len(op_call_args):]:
            if k in n.call_kwargs:
                tensor_input_alloc.append(k)

    return tensor_input_alloc

def _extract_const_attrs_from_args(op_call_args: List[Any],
                                   op_call_kwargs: Dict[str, Any],
                                   inputs_as_list: bool,
                                   tensor_inputs_alloc: List,
                                   weights: Dict[Union[str, int], Any]) -> Tuple:
    """
    Extract const weights of the layer from the operator's arguments list.
    This function extracts the attributes, updates the nodes weights dictionary and removes them from the original
    arguments list.

    Args:
        op_call_args: A list of the operator arguments.
        op_call_kwargs: A mapping of key-arguments of the operator.
        inputs_as_list: Whether the input of the layer is a list.
        tensor_inputs_alloc: Allocation of argument inputs to the operator (if there are const inputs, otherwise None).
        weights: Node weights mapping. This dictionary is modified by this function.

    Returns: A modified operator arguments list.

    """

    move_args_to_kwargs = tensor_inputs_alloc is not None and len(tensor_inputs_alloc) > 0

    # read weights from call args
    for i, arg in enumerate(op_call_args[0] if inputs_as_list else op_call_args):
        if is_const(arg):
            weights.update({i: to_numpy(arg, is_single_tensor=True)})
        else:
            if not inputs_as_list:
                if move_args_to_kwargs:
                    # In this case we move all arguments and inputs to the kwargs
                    op_call_kwargs.update({tensor_inputs_alloc[i]: arg})

    # remove weights and KerasTensors from op_call_args
    if inputs_as_list:
        op_call_args = tuple(op_call_args[1:])
    else:
        op_call_args = tuple([a for i, a in enumerate(op_call_args)
                              if not (i in weights or is_tensor(a) or (move_args_to_kwargs and tensor_inputs_alloc[i]
                                                                       in op_call_kwargs))])

    return op_call_args


def _has_const_attributes(op_call_args: List, op_call_kwargs: Dict, input_as_list: bool) -> bool:
    """
    Returns whether the layer's input include a constant tensor (that we might want to quantize).

    Args:
        op_call_args: A list of arguments to the layer.
        op_call_kwargs: A dictionary of key-arguments to the layer.
        input_as_list: Whether the input to the layer is a list of tensors.

    Returns: True if the input arguments include a constant tensor, False otherwise.

    """
    if input_as_list:
        return any([is_const(a) for a in op_call_args[0]])
    const_args = [a for a in op_call_args if is_const(a)]
    const_kwargs = [k for k, v in op_call_kwargs.items() if is_const(v)]

    return len(const_args) > 0 or len(const_kwargs) > 0


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
    op_call_args = copy(node.call_args)
    op_call_kwargs = copy(node.call_kwargs)
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
        # Some functional ops should receive the input tensors as a list,
        # so each FunctionalNode holds a flag to indicate that.
        # Other functional ops can receive each argument as list, but in that case not all inputs appear in that list.
        inputs_as_list = __is_functional_inputs_a_list(op_call_args, keras_layer)

        kwarg2index = get_kwargs2index(keras_layer)

        # Functional nodes do not have weights, but may have constants in their call_args and\or
        # call kwargs. Therefore, we extract these constants and save them in the node's weights as
        # positional weights. Positional weights are weights whose keys are the index of that constant
        # in the call_args.
        # All KerasTensor and positional weights are removed from the call_args\kwargs. They are restored
        # in the model builder.
        if len(weights) > 0:
            Logger.critical('Functional nodes are not expected to have weights in this framework.')

        # Build tensor_input_alloc required for the model builder. All inputs are received as a list in the builder,
        # so tensor_input_alloc is used to allocate each input in the correct place in the node's args & kwargs.
        tensor_input_alloc = None if not _has_const_attributes(op_call_args, op_call_kwargs, inputs_as_list) \
            else _build_arguments_alloc(node, inputs_as_list, kwarg2index)

        op_call_args = _extract_const_attrs_from_args(op_call_args, op_call_kwargs, inputs_as_list,
                                                      tensor_input_alloc, weights)
        op_call_kwargs = _extract_const_attrs_from_kwargs(op_call_kwargs, kwarg2index, weights)

        node = FunctionalNode(node_name,
                              layer_config,
                              input_shape,
                              output_shape,
                              weights,
                              layer_class,
                              op_call_args,
                              op_call_kwargs,
                              is_reused,
                              reuse_group,
                              functional_op=keras_layer.function,
                              inputs_as_list=inputs_as_list,
                              tensor_input_allocs=tensor_input_alloc)
    else:
        # Read constant weights from layers such as layers.Add
        if len(op_call_args) > 0 and isinstance(op_call_args[0], (list, tuple)):
            for i, arg in enumerate(op_call_args[0]):
                if is_const(arg):
                    weights.update({i: to_numpy(arg, is_single_tensor=True)})

        node = BaseNode(node_name,
                        layer_config,
                        input_shape,
                        output_shape,
                        weights,
                        layer_class,
                        is_reused,
                        reuse_group,
                        is_custom=is_keras_custom_layer(layer_class))

    node_name_to_node[node_name] = node
    return node


def __is_functional_inputs_a_list(op_call_args: Any, keras_layer: Any) -> bool:
    """
    Check whether the input tensors should be passed as a list
    or not. This is relevant only for specific TF operators that are specified in the function's condition.

    Args:
        op_call_args: Arguments list to check.
        keras_layer: TFOpLambda layer.

    Returns:
        Whether the input tensors should be passed as a list or not.
    """

    return (keras_layer.symbol in
            [TFOpLambda(tf.concat).symbol, TFOpLambda(tf.stack).symbol,TFOpLambda(tf.add_n).symbol] and
            len(op_call_args) > 0 and
            isinstance(op_call_args[0], list))
