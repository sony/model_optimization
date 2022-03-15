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
    from tensorflow.keras.layers import Input
    from tensorflow.python.keras.layers.core import TFOpLambda
    from tensorflow.python.keras.engine.base_layer import TensorFlowOpLayer
    from tensorflow.python.keras.layers import Layer
else:
    from keras import Input
    from keras.layers.core import TFOpLambda
    from keras.engine.base_layer import TensorFlowOpLayer, Layer

from model_compression_toolkit.common.model_builder_mode import ModelBuilderMode
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from typing import Tuple, Any, Dict, List
from tensorflow.python.util.object_identity import Reference as TFReference
from model_compression_toolkit.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.common.logger import Logger
from model_compression_toolkit import common
from model_compression_toolkit.common.gptq.gptq_config import GradientPTQConfig
from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.keras.quantizer.mixed_precision.quantization_config_factory import \
    quantization_config_builder_mixed_precision
from model_compression_toolkit.keras.quantizer.gradient_ptq.config_factory import quantization_config_builder_gptq
from model_compression_toolkit.common import BaseNode, Graph
from model_compression_toolkit.common.graph.edge import EDGE_SINK_INDEX
from model_compression_toolkit.keras.back2framework.instance_builder import OperationHandler
from model_compression_toolkit.keras.reader.connectivity_handler import OutTensor

# In tf2.3 fake quant node is implemented as TensorFlowOpLayer, while in tf2.4 as TFOpLambda.
FQ_NODE_OP_V2_3 = 'FakeQuantWithMinMaxVars'
FQ_NODE_OP_V2_4 = 'quantization.fake_quant_with_min_max_vars'
BATCH_INPUT_SHAPE = 'batch_input_shape'


def get_node_name_from_layer(layer: Layer) -> str:
    """
    Get a node's name from the layer it was built from. For TensorFlowOpLayer
    we remove the prefix "tf_op_layer".

    Args:
        layer: Keras Layer to get its corresponding node's name.

    Returns:
        Name of the node that was built from the passed layer.
    """

    name = layer.name
    if isinstance(layer, TensorFlowOpLayer):  # remove TF op layer prefix
        name = '_'.join(name.split('_')[3:])
    return name


def is_layer_fake_quant(layer: Layer) -> bool:
    """
    Check whether a Keras layer is a fake quantization layer or not.
    Args:
        layer: Layer to check if it's a fake quantization layer or not.

    Returns:
        Whether a Keras layer is a fake quantization layer or not.
    """
    # in tf2.3 fake quant node is implemented as TensorFlowOpLayer, while in tf2.4 as TFOpLambda
    return (isinstance(layer, TensorFlowOpLayer) and layer.node_def.op == FQ_NODE_OP_V2_3) or (
            isinstance(layer, TFOpLambda) and layer.symbol == FQ_NODE_OP_V2_4)


def build_input_tensors_list(node: BaseNode,
                             graph: Graph,
                             node_to_output_tensors_dict: Dict[BaseNode, List[TFReference]]) -> List[List[TFReference]]:
    """
    Given a node, build a list of input tensors the node gets. The list is built
    based on the node's incoming edges and previous nodes' output tensors.

    Args:
        node: Node to build its input tensors list.
        graph: Graph the node is in.
        node_to_output_tensors_dict: A dictionary from a node to its output tensors.

    Returns:
        A list of the node's input tensors.
    """

    input_tensors = []
    # Go over a sorted list of the node's incoming edges, and for each source node get its output tensors.
    # Append them in a result list.
    for ie in graph.incoming_edges(node, sort_by_attr=EDGE_SINK_INDEX):
        _input_tensors = [node_to_output_tensors_dict[ie.source_node][ie.source_index]]
        input_tensors.append(_input_tensors)
    return input_tensors


def run_operation(n: BaseNode,
                  input_tensors: List[List[TFReference]],
                  op_func: Layer,
                  input_nodes_to_input_tensors: Dict[BaseNode, Any],
                  mode: ModelBuilderMode = ModelBuilderMode.QUANTIZED) -> List[TFReference]:
    """
    Applying the layer (op_func) to the input tensors (input_tensors).
    If quantized is set to True, and the layer's corresponding node (n) has quantization
    attributes, an additional fake-quantization node is built and appended to the layer.

    Args:
        n: The corresponding node of the layer it runs.
        input_tensors: List of references to Keras tensors that are the layer's inputs.
        op_func: Layer to apply to the input tensors.
        input_nodes_to_input_tensors: A dictionary from a node to its input tensors.
        mode: model quantization mode from ModelBuilderMode

    Returns:
        A list of references to Keras tensors. The layer's output tensors after applying the
        layer to the input tensors.
    """

    if len(input_tensors) == 0:  # Placeholder handling
        out_tensors_of_n_float = input_nodes_to_input_tensors[n]
        out_tensors_of_n = out_tensors_of_n_float
        if n.is_activation_quantization_enabled():
            if mode in [ModelBuilderMode.QUANTIZED, ModelBuilderMode.GPTQ] and n.final_activation_quantization_cfg:
                # Adding a fake quant node to Input when in GPTQ mode because quantize_model doesn't quantize the
                # input layer
                out_tensors_of_n = n.final_activation_quantization_cfg.quantize_node_output(out_tensors_of_n_float)
            elif mode in [ModelBuilderMode.MIXEDPRECISION]:
                # TODO: refactor after implementing activations mixed precision
                assert n.is_all_activation_candidates_equal()
                out_tensors_of_n = n.candidates_quantization_cfg[0].activation_quantization_cfg.quantize_node_output(
                    out_tensors_of_n_float)

    else:
        input_tensors = [tensor for tensor_list in input_tensors for tensor in tensor_list]  # flat list of lists
        # Build a functional node using its args
        if isinstance(n, FunctionalNode):
            if n.inputs_as_list:  # If the first argument should be a list of tensors:
                out_tensors_of_n_float = op_func(input_tensors, *n.op_call_args, **n.op_call_kwargs)
            else:  # If the input tensors should not be a list but iterated:
                out_tensors_of_n_float = op_func(*input_tensors, *n.op_call_args, **n.op_call_kwargs)
        else:
            # If operator expects a single input tensor, it cannot be a list as it should
            # have a dtype field.
            if len(input_tensors) == 1:
                input_tensors = input_tensors[0]
            out_tensors_of_n_float = op_func(input_tensors)
        out_tensors_of_n = out_tensors_of_n_float

        # Add a fake quant node if the node has an activation threshold.
        if n.is_activation_quantization_enabled():
            if mode in [ModelBuilderMode.QUANTIZED, ModelBuilderMode.GPTQ] and n.final_activation_quantization_cfg:
                out_tensors_of_n = n.final_activation_quantization_cfg.quantize_node_output(out_tensors_of_n_float)
            elif mode in [ModelBuilderMode.MIXEDPRECISION]:
                # TODO: refactor after implementing activations mixed precision
                assert n.is_all_activation_candidates_equal()
                out_tensors_of_n = n.candidates_quantization_cfg[0].activation_quantization_cfg.quantize_node_output(
                    out_tensors_of_n_float)

    return out_tensors_of_n, out_tensors_of_n_float


def convert_node2name(in_node_to_output_tensors_dict):
    node_name_to_outtensors = dict()
    for node, tensors in in_node_to_output_tensors_dict.items():
        node_name_to_outtensors[node.name] = tensors
    return node_name_to_outtensors


def model_builder(graph: common.Graph,
                  mode: ModelBuilderMode = ModelBuilderMode.QUANTIZED,
                  append2output=None,
                  fw_info: FrameworkInfo = DEFAULT_KERAS_INFO, gptq_config: GradientPTQConfig = None) -> Tuple[
    tf.keras.models.Model, Any]:
    """
    Build a Keras model from a graph representing the model.
    The model is built by converting the graph nodes to Keras layers and applying them sequentially to get the model
    output tensors. The output tensors list and an input tensors list, then use to build the model.
    When the model is not built in float mode, the graph is being transformed by additional substitutions.

    Args:
        graph: Graph to build its corresponding Keras model.
        mode: Building mode. Read ModelBuilderMode description for more info.
        append2output: List of nodes or OutTensor objects. In float building mode,
        when the list contains nodes, all output tensors of all nodes are set as the model outputs.
        fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).
        This is needed when using MIXEDPRECISION or GPTQ mode for passing the kernel attributes to
        the QuanteWrapper we use in both of these cases.
        gptq_config: GPTQ Configuration class..

    Returns:
        A tuple of the model, and an UserInformation object.
    """
    if gptq_config is None and mode == ModelBuilderMode.GPTQ:
        Logger.exception("Building a model in GPTQ require GPTQ configuration as input")
    node_to_output_tensors_dict = dict()
    node_to_output_tensors_dict_float = dict()
    model_output_tensors = []

    # Build an OperationHandler to handle conversions from graph nodes to Keras operators.
    oh = OperationHandler(graph)

    # Create a list of output nodes with their tensors' indices that are model's output. When building
    # in float mode, if a node has multiple out tensors, the node should appear in append2output multiple
    # times, thus creating OutTensor for each output tensor.
    if append2output is not None:
        output_list = [OutTensor(n, 0) for n in append2output]
    else:
        output_list = graph.get_outputs()

    # Hold a dictionary from an input node to its corresponding input tensor. It is needed for when
    # building the model. Initially input nodes with input tensors are added to the dictionary,
    # as they're not added later.
    input_nodes_to_input_tensors = {inode: Input(inode.framework_attr[BATCH_INPUT_SHAPE][1:]) for
                                    inode in graph.get_inputs()}

    # Build a list of the model's input tensors. Switching from a dictionary to a list
    # to keep the tensors input order, since inputs in Graph are ordered by their indices.
    inputs_list = []
    for input_node in graph.get_inputs():
        inputs_list.append(input_nodes_to_input_tensors.get(input_node))

    # Build a dictionary from node to its output tensors, by applying the layers sequentially.
    for n in oh.node_sort:
        op_func = oh.get_node_op_function(n)  # Get node operation function
        input_tensors = build_input_tensors_list(n,
                                                 graph,
                                                 node_to_output_tensors_dict)  # Fetch Node inputs
        out_tensors_of_n, out_tensors_of_n_float = run_operation(n,  # Run node operation and fetch outputs
                                                                 input_tensors,
                                                                 op_func,
                                                                 input_nodes_to_input_tensors,
                                                                 mode)

        if isinstance(out_tensors_of_n, list):
            node_to_output_tensors_dict.update({n: out_tensors_of_n})
            node_to_output_tensors_dict_float.update({n: out_tensors_of_n_float})
        else:
            node_to_output_tensors_dict.update({n: [out_tensors_of_n]})
            node_to_output_tensors_dict_float.update({n: [out_tensors_of_n_float]})

    # convert node_to_output_tensors_dict keys to nodes' names since oh.node_sort contains different objects than
    # original graph nodes.
    node_name_to_outtensors = convert_node2name(node_to_output_tensors_dict)
    node_name_to_outtensors_float = convert_node2name(node_to_output_tensors_dict_float)

    for ot in output_list:
        if len(node_name_to_outtensors[ot.node.name]) == 1 or append2output is None:
            if mode == ModelBuilderMode.GPTQ:
                model_output_tensors.append(node_name_to_outtensors_float[ot.node.name][ot.node_out_index])
            else:
                model_output_tensors.append(node_name_to_outtensors[ot.node.name][ot.node_out_index])
        else:  # When building float model - we collect all outputs from all nodes regardless the actual model's outputs
            if mode == ModelBuilderMode.GPTQ:
                # In case of GPTQ output the float data for the loss and not quantized.
                model_output_tensors.append(node_name_to_outtensors_float[ot.node.name])
            else:
                model_output_tensors.append(node_name_to_outtensors[ot.node.name])

    # Build the model.
    model = tf.keras.Model(inputs=inputs_list, outputs=model_output_tensors)

    # In GPTQ mode, wrap each layer in a QuantizeWrapper containing QuantizeConfig
    # that's built using the node quantization attributes.
    if mode == ModelBuilderMode.GPTQ:
        def _quantize(layer):
            nodes = graph.find_node_by_name(get_node_name_from_layer(layer))
            if len(nodes) == 1:
                node = nodes[0]
                return QuantizeWrapper(layer, quantization_config_builder_gptq(node, fw_info, gptq_config))
            elif is_layer_fake_quant(layer):
                return layer
            else:
                raise Exception(
                    f"Mismatch between keras model and graph can't find node named: {get_node_name_from_layer(layer)}")

        # clone each layer in the model and apply _quantize to the layer.
        model = tf.keras.models.clone_model(model, input_tensors=None, clone_function=_quantize)

    # In MIXEDPRECISION mode, wrap each layer that can be configured with bitwidth
    # in a QuantizeWrapper containing QuantizeConfig that holds a quantizer that
    # stores the quantized weights using all possible bitwidths.
    elif mode == ModelBuilderMode.MIXEDPRECISION:
        def _quantize_multiple_nbits(layer):
            nodes = graph.find_node_by_name(get_node_name_from_layer(layer))
            if len(nodes) == 1:
                node = nodes[0]
                # Wrap only if its weights should be quantized
                if node.is_weights_quantization_enabled():
                    return QuantizeWrapper(layer, quantization_config_builder_mixed_precision(node, fw_info))
                return layer

            elif is_layer_fake_quant(layer):
                return layer
            else:
                raise Exception(
                    f'Mismatch between keras model and graph cant find node named: {get_node_name_from_layer(layer)}')

        # clone each layer in the model and apply _quantize to the layer.
        model = tf.keras.models.clone_model(model, input_tensors=None, clone_function=_quantize_multiple_nbits)

    # Models that were built in float or quantized mode, should not be modified anymore.
    elif mode == ModelBuilderMode.FLOAT or mode == ModelBuilderMode.QUANTIZED:
        pass
    else:
        common.Logger.exception(f'Unknown model mode: {mode}')

    return model, graph.user_info
