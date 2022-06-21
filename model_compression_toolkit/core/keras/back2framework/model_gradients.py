# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
import numpy as np
import tensorflow as tf

# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.
from keras.layers import Softmax

if tf.__version__ < "2.6":
    from tensorflow.keras.layers import Softmax
    from tensorflow.python.keras.layers.core import TFOpLambda
    from tensorflow.python.keras.engine.base_layer import TensorFlowOpLayer
    from tensorflow.python.keras.layers import Layer
else:
    from keras.layers import Softmax
    from keras.layers.core import TFOpLambda
    from keras.engine.base_layer import TensorFlowOpLayer, Layer

from typing import Any, Dict, List
from tensorflow.python.util.object_identity import Reference as TFReference
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.graph.edge import EDGE_SINK_INDEX
from model_compression_toolkit.core.keras.back2framework.instance_builder import OperationHandler

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
                  input_nodes_to_input_tensors: Dict[BaseNode, Any]) -> List[TFReference]:
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
        # raise Exception("Need to build from datas")
        out_tensors_of_n = input_nodes_to_input_tensors[n]  # Check if cast is need
        # out_tensors_of_n
        # out_tensors_of_n_float = input_nodes_to_input_tensors[n]
        # out_tensors_of_n = out_tensors_of_n_float
        # if n.is_activation_quantization_enabled():
        #     if mode in [ModelBuilderMode.QUANTIZED, ModelBuilderMode.GPTQ] and n.final_activation_quantization_cfg:
        #         # Adding a fake quant node to Input when in GPTQ mode because quantize_model doesn't quantize the
        #         # input layer
        #         out_tensors_of_n = n.final_activation_quantization_cfg.quantize_node_output(out_tensors_of_n_float)
        #     elif mode in [ModelBuilderMode.MIXEDPRECISION]:
        #         if n.is_all_activation_candidates_equal():
        #             # otherwise, we want to use the float tensor when building the model for MP search
        #             out_tensors_of_n = n.candidates_quantization_cfg[
        #                 0].activation_quantization_cfg.quantize_node_output(out_tensors_of_n_float)

    else:
        input_tensors = [tensor for tensor_list in input_tensors for tensor in tensor_list]  # flat list of lists
        # Build a functional node using its args
        if isinstance(n, FunctionalNode):
            if n.inputs_as_list:  # If the first argument should be a list of tensors:
                out_tensors_of_n = op_func(input_tensors, *n.op_call_args, **n.op_call_kwargs)
            else:  # If the input tensors should not be a list but iterated:
                out_tensors_of_n = op_func(*input_tensors, *n.op_call_args, **n.op_call_kwargs)
        else:
            # If operator expects a single input tensor, it cannot be a list as it should
            # have a dtype field.
            if len(input_tensors) == 1:
                input_tensors = input_tensors[0]
            out_tensors_of_n = op_func(input_tensors)

    return out_tensors_of_n


def convert_node2name(in_node_to_output_tensors_dict):
    node_name_to_outtensors = dict()
    for node, tensors in in_node_to_output_tensors_dict.items():
        node_name_to_outtensors[node.name] = tensors
    return node_name_to_outtensors


def model_grad(graph_float: common.Graph,
               model_input_tensors: Dict[BaseNode, np.ndarray],
               intresent_points,
               output_list):
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
    node_to_output_tensors_dict = dict()

    # Build an OperationHandler to handle conversions from graph nodes to Keras operators.
    oh = OperationHandler(graph_float)

    input_nodes_to_input_tensors = {inode: tf.convert_to_tensor(model_input_tensors[inode]) for
                                    inode in graph_float.get_inputs()}  # Cast numpy array to tf.Tensor

    # for ip in intresent_points:
    intresent_points_tensors = []
    output_tensors = []
    with tf.GradientTape(persistent=True) as g:
        # Build a dictionary from node to its output tensors, by applying the layers sequentially.
        for n in oh.node_sort:
            op_func = oh.get_node_op_function(n)  # Get node operation function
            if 'argmax' in n.name:
                def softargmax(x, axis):
                    beta = 1
                    x_range = tf.range(x.shape.as_list()[axis], dtype=x.dtype)
                    return tf.reduce_sum(Softmax(axis=axis)(x * beta) * x_range, axis=axis)

                op_func = softargmax

            input_tensors = build_input_tensors_list(n,
                                                     graph_float,
                                                     node_to_output_tensors_dict)  # Fetch Node inputs
            out_tensors_of_n = run_operation(n,  # Run node operation and fetch outputs
                                             input_tensors,
                                             op_func,
                                             input_nodes_to_input_tensors)

            if n in intresent_points:
                g.watch(out_tensors_of_n)
                intresent_points_tensors.append(out_tensors_of_n)
            if n in output_list:
                output_tensors.append(out_tensors_of_n)

            if isinstance(out_tensors_of_n, list):
                node_to_output_tensors_dict.update({n: out_tensors_of_n})
            else:
                node_to_output_tensors_dict.update({n: [out_tensors_of_n]})
        output_sum = 0
        for output in output_tensors:
            output_sum += tf.reduce_sum(output)
    ###########################################
    # Compute Gradients
    ##########################################
    ipt_grad_score = []
    for ipt in intresent_points_tensors:
        # output_sum = 0
        # hessian_trace_aprrox = []
        # for output in output_tensors:
        #     # output_sum += tf.reduce_sum(output)
        #
        #     grad_ipt = g.gradient(output, ipt)
        #     # grad_ipt = g.gradient(output_sum, ipt)
        #     if grad_ipt is None:
        #         continue
        #     # hessian_trace_aprrox.append(tf.reduce_sum(tf.pow(grad_ipt, 2.0)))
        #     hessian_trace_aprrox.append(tf.reduce_sum(grad_ipt))
        # ipt_grad_score.append(tf.reduce_mean(hessian_trace_aprrox))

        grad_ipt = g.gradient(output_sum, ipt)
        # hessian_trace_aprrox = tf.reduce_sum(tf.abs(grad_ipt))
        hessian_trace_aprrox = tf.reduce_sum(tf.pow(grad_ipt, 2.0))
        ipt_grad_score.append(hessian_trace_aprrox)

    # return ipt_grad_score
    # max_grad_score = max(ipt_grad_score[:-1]).numpy()

    ###########################
    # Print gradient score order
    sorted_inds = np.flip(np.array(ipt_grad_score).argsort())
    sorted_layers = {intresent_points[i].name: ipt_grad_score[i].numpy() for i in sorted_inds}
    print(sorted_layers)
    ##########################

    max_idx = np.argmax(ipt_grad_score)
    second_max_idx = np.argmax(ipt_grad_score[:max_idx] + ipt_grad_score[max_idx + 1:])

    if (ipt_grad_score[max_idx] / ipt_grad_score[second_max_idx]).numpy() >= 1e3:
        max_grad_score = ipt_grad_score[second_max_idx].numpy()
        ipt_grad_score[max_idx] = ipt_grad_score[second_max_idx]
    else:
        max_grad_score = ipt_grad_score[max_idx].numpy()

    return [s.numpy() / max_grad_score for s in ipt_grad_score]
    # return [s.numpy() for s in ipt_grad_score]