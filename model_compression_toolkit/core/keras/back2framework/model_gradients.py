# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from packaging import version

# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.
from tqdm import tqdm

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.engine.base_layer import Layer
else:
    from tensorflow.python.keras.engine.base_layer import Layer

from typing import Any, Dict, List, Tuple
from tensorflow.python.util.object_identity import Reference as TFReference
from model_compression_toolkit.constants import EPS, MIN_JACOBIANS_ITER, JACOBIANS_COMP_TOLERANCE
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.graph.edge import EDGE_SINK_INDEX
from model_compression_toolkit.core.keras.back2framework.instance_builder import OperationHandler
from model_compression_toolkit.logger import Logger


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

    Args:
        n: The corresponding node of the layer it runs.
        input_tensors: List of references to Keras tensors that are the layer's inputs.
        op_func: Layer to apply to the input tensors.
        input_nodes_to_input_tensors: A dictionary from a node to its input tensors.

    Returns:
        A list of references to Keras tensors. The layer's output tensors after applying the
        layer to the input tensors.
    """

    if len(input_tensors) == 0:  # Placeholder handling
        out_tensors_of_n = input_nodes_to_input_tensors[n]
    else:
        input_tensors = [tensor for tensor_list in input_tensors for tensor in tensor_list]  # flat list of lists
        # Build a functional node using its args
        if isinstance(n, FunctionalNode):
            if n.inputs_as_list:  # If the first argument should be a list of tensors:
                out_tensors_of_n = op_func(input_tensors, *n.op_call_args, **n.op_call_kwargs)
            else:  # If the input tensors should not be a list but iterated:
                out_tensors_of_n = op_func(*input_tensors, *n.op_call_args, **n.op_call_kwargs)
        else:
            # If operator expects a single input tensor, it cannot be a list as it should have a dtype field.
            if len(input_tensors) == 1:
                input_tensors = input_tensors[0]
            out_tensors_of_n = op_func(input_tensors)

    return out_tensors_of_n


def keras_iterative_approx_jacobian_trace(graph_float: common.Graph,
                                          model_input_tensors: Dict[BaseNode, np.ndarray],
                                          interest_points: List[BaseNode],
                                          output_list: List[BaseNode],
                                          all_outputs_indices: List[int],
                                          alpha: float = 0.3,
                                          n_iter: int = 50,
                                          norm_weights: bool = True) -> List[float]:
    """
    Computes an approximation of the power of the Jacobian trace of a Keras model's outputs with respect to the feature maps of
    the set of given interest points. It then uses the power of the Jacobian trace for each interest point and normalized the
    values, to be used as weights for weighted average in distance metric computation.

    Args:
        graph_float: Graph to build its corresponding Keras model.
        model_input_tensors: A mapping between model input nodes to an input batch.
        interest_points: List of nodes which we want to get their feature map as output, to calculate distance metric.
        output_list: List of nodes that considered as model's output for the purpose of gradients computation.
        all_outputs_indices: Indices of the model outputs and outputs replacements (if exists),
            in a topological sorted interest points list.
        alpha: A tuning parameter to allow calibration between the contribution of the output feature maps returned
            weights and the other feature maps weights (since the gradient of the output layers does not provide a
            compatible weight for the distance metric computation).
        n_iter: The number of random iterations to calculate the approximated power of the Jacobian trace for each interest point.
        norm_weights: Whether to normalize the returned weights (to get values between 0 and 1).

    Returns: A list of (possibly normalized) jacobian-based weights to be considered as the relevancy that each interest
    point's output has on the model's output.
    """

    if len(interest_points) == 1:
        # Only one compare point, nothing else to "weight"
        return [1.0]

    if not all([images.shape[0] == 1 for node, images in model_input_tensors.items()]):
        Logger.critical("Iterative jacobian trace computation is only supported on a single image sample")  # pragma: no cover

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
        outputs, interest_points_tensors = _model_outputs_computation(graph_float,
                                                                      model_input_tensors,
                                                                      interest_points,
                                                                      output_list,
                                                                      gradient_tape=g)

        # Concat outputs
        # First, we need to unfold all outputs that are given as list, to extract the actual output tensors
        unfold_outputs = []
        for output in outputs:
            if isinstance(output, List):
                unfold_outputs += output
            else:
                unfold_outputs.append(output)

        r_outputs = [tf.reshape(output, shape=[output.shape[0], -1]) for output in unfold_outputs]

        concat_axis_dim = [o.shape[0] for o in r_outputs]
        if not all(d == concat_axis_dim[0] for d in concat_axis_dim):
            Logger.critical("Can't concat model's outputs for gradients calculation since the shape of the first axis "  # pragma: no cover
                            "is not equal in all outputs.")

        output = tf.concat(r_outputs, axis=1)

        ipts_jac_trace_approx = []
        for ipt in tqdm(interest_points_tensors):  # Per Interest point activation tensor
            trace_jv = []
            for j in range(n_iter):  # Approximation iterations
                # Getting a random vector with normal distribution
                v = tf.random.normal(shape=output.shape)
                f_v = tf.reduce_sum(v * output)

                with g.stop_recording():
                    # Computing the jacobian approximation by getting the gradient of (output * v)
                    jac_v = g.gradient(f_v, ipt, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                    jac_v = tf.reshape(jac_v, [jac_v.shape[0], -1])
                    jac_trace_approx = tf.reduce_mean(tf.reduce_sum(tf.pow(jac_v, 2.0)))

                    # If the change to the mean Jacobian approximation is insignificant we stop the calculation
                    if j > MIN_JACOBIANS_ITER:
                        new_mean = np.mean([jac_trace_approx, *trace_jv])
                        delta = new_mean - np.mean(trace_jv)
                        if np.abs(delta) / (np.abs(new_mean) + 1e-6) < JACOBIANS_COMP_TOLERANCE:
                            trace_jv.append(jac_trace_approx)
                            break

                    trace_jv.append(jac_trace_approx)
            ipts_jac_trace_approx.append(2 * tf.reduce_mean(trace_jv) / output.shape[-1])  # Get averaged squared jacobian trace approximation

        ipts_jac_trace_approx = tf.reduce_mean([ipts_jac_trace_approx], axis=0)  # Just to get one tensor instead of list of tensors with single element

        if norm_weights:
            return _normalize_weights(ipts_jac_trace_approx, all_outputs_indices, alpha)
        else:
            return ipts_jac_trace_approx


def _model_outputs_computation(graph_float: common.Graph,
                               model_input_tensors: Dict[BaseNode, np.ndarray],
                               interest_points:  List[BaseNode],
                               output_list: List[BaseNode],
                               gradient_tape: tf.GradientTape) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
    """
    Computes the model's output according to the given graph representation on the given input,
    while recording necessary intermediate tensors for gradients computation.

    Args:
        graph_float: Graph to build its corresponding Keras model.
        model_input_tensors: A mapping between model input nodes to an input batch.
        interest_points: List of nodes which we want to get their feature map as output, to calculate distance metric.
        output_list: List of nodes that considered as model's output for the purpose of gradients computation.
        gradient_tape: A GradientTape object for recording necessary info for computing gradients.

    Returns: A list of output tensors and a list of activation tensors of all interest points.

    """

    node_to_output_tensors_dict = dict()

    # Build an OperationHandler to handle conversions from graph nodes to Keras operators.
    oh = OperationHandler(graph_float)
    input_nodes_to_input_tensors = {inode: tf.convert_to_tensor(model_input_tensors[inode]) for
                                    inode in graph_float.get_inputs()}  # Cast numpy array to tf.Tensor

    interest_points_tensors = []
    output_tensors = []
    for n in oh.node_sort:
        # Build a dictionary from node to its output tensors, by applying the layers sequentially.
        op_func = oh.get_node_op_function(n)  # Get node operation function

        input_tensors = build_input_tensors_list(n,
                                                 graph_float,
                                                 node_to_output_tensors_dict)  # Fetch Node inputs
        out_tensors_of_n = run_operation(n,  # Run node operation and fetch outputs
                                         input_tensors,
                                         op_func,
                                         input_nodes_to_input_tensors)

        # Gradients can be computed only on float32 tensors
        if isinstance(out_tensors_of_n, list):
            for i, t in enumerate(out_tensors_of_n):
                out_tensors_of_n[i] = tf.dtypes.cast(t, tf.float32)
        else:
            out_tensors_of_n = tf.dtypes.cast(out_tensors_of_n, tf.float32)

        if n in interest_points:
            # Recording the relevant feature maps onto the gradient tape
            gradient_tape.watch(out_tensors_of_n)
            interest_points_tensors.append(out_tensors_of_n)
        if n in output_list:
            output_tensors.append(out_tensors_of_n)

        if isinstance(out_tensors_of_n, list):
            node_to_output_tensors_dict.update({n: out_tensors_of_n})
        else:
            node_to_output_tensors_dict.update({n: [out_tensors_of_n]})

    return output_tensors, interest_points_tensors


def _normalize_weights(jacobians_traces: List,
                       all_outputs_indices: List[int],
                       alpha: float) -> List[float]:
    """
    Output layers or layers that come after the model's considered output layers,
    are assigned with a constant normalized value, according to the given alpha variable and the number of such layers.
    Other layers returned weights are normalized by dividing the jacobian-based weights value by the sum of all other values.

    Args:
        jacobians_traces: The approximated average jacobian-based weights of each interest point.
        all_outputs_indices: A list of indices of all nodes that consider outputs.
        alpha: A multiplication factor.

    Returns: Normalized list of jacobian-based weights (for each interest point).

    """

    sum_without_outputs = sum([jacobians_traces[i] for i in range(len(jacobians_traces)) if i not in all_outputs_indices])
    normalized_grads_weights = [_get_normalized_weight(grad, i, sum_without_outputs, all_outputs_indices, alpha)
                                for i, grad in enumerate(jacobians_traces)]

    return normalized_grads_weights


def _get_normalized_weight(grad: float,
                           i: int,
                           sum_without_outputs: float,
                           all_outputs_indices: List[int],
                           alpha: float) -> float:
    """
    Normalizes the node's gradient value. If it is an output or output replacement node than the normalized value is
    a constant, otherwise, it is normalized by dividing with the sum of all gradient values.

    Args:
        grad: The gradient value.
        i: The index of the node in the sorted interest points list.
        sum_without_outputs: The sum of all gradients of nodes that are not considered outputs.
        all_outputs_indices: A list of indices of all nodes that consider outputs.
        alpha: A multiplication factor.

    Returns: A normalized jacobian-based weights.

    """

    if i in all_outputs_indices:
        return alpha / len(all_outputs_indices)
    else:
        return ((1 - alpha) * grad / (sum_without_outputs + EPS)).numpy()
