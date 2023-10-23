# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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

from typing import List, Tuple, Dict, Any

import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from tqdm import tqdm
import numpy as np

from model_compression_toolkit.constants import MIN_JACOBIANS_ITER, JACOBIANS_COMP_TOLERANCE, EPS, \
    HESSIAN_NUM_ITERATIONS
from model_compression_toolkit.core.common.graph.edge import EDGE_SINK_INDEX
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.common.hessian import TraceHessianRequest, HessianInfoGranularity
from model_compression_toolkit.core.keras.back2framework.instance_builder import OperationHandler
from model_compression_toolkit.core.keras.hessian.trace_hessian_calculator_keras import TraceHessianCalculatorKeras
from model_compression_toolkit.logger import Logger
from tensorflow.python.util.object_identity import Reference as TFReference


class ActivationTraceHessianCalculatorKeras(TraceHessianCalculatorKeras):
    """
    Keras implementation of the Trace Hessian Calculator for activations.
    """
    def __init__(self,
                 graph: Graph,
                 input_images: List[tf.Tensor],
                 fw_impl,
                 trace_hessian_request: TraceHessianRequest,
                 num_iterations_for_approximation: int = HESSIAN_NUM_ITERATIONS):
        """
        Args:
            graph: Computational graph for the float model.
            input_images: List of input images for the computation.
            fw_impl: Framework-specific implementation for trace Hessian approximation computation.
            trace_hessian_request: Configuration request for which to compute the trace Hessian approximation.
            num_iterations_for_approximation: Number of iterations to use when approximating the Hessian trace.

        """
        super(ActivationTraceHessianCalculatorKeras, self).__init__(graph=graph,
                                                                    input_images=input_images,
                                                                    fw_impl=fw_impl,
                                                                    trace_hessian_request=trace_hessian_request,
                                                                    num_iterations_for_approximation=num_iterations_for_approximation)

    def compute(self) -> List[float]:
        """
        Compute the approximation of the trace of the Hessian w.r.t a node's activations.

        Returns:
            List[float]: Approximated trace of the Hessian for an interest point.
        """
        if self.hessian_request.granularity == HessianInfoGranularity.PER_TENSOR:
            output_list = self._get_model_output_replacement()

            # Record operations for automatic differentiation
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
                outputs, interest_points_tensors = self._get_model_outputs_for_single_image(output_list,
                                                                                            gradient_tape=g)

                # Unfold and concatenate all outputs to form a single tensor
                output = self._concat_tensors(outputs)

                # List to store the approximated trace of the Hessian for each interest point
                trace_approx_by_node = []
                # Loop through each interest point activation tensor
                for ipt in tqdm(interest_points_tensors):  # Per Interest point activation tensor
                    interest_point_scores = [] # List to store scores for each interest point
                    for j in range(self.num_iterations_for_approximation):  # Approximation iterations
                        # Getting a random vector with normal distribution
                        v = tf.random.normal(shape=output.shape)
                        f_v = tf.reduce_sum(v * output)

                        with g.stop_recording():
                            # Computing the approximation by getting the gradient of (output * v)
                            gradients = g.gradient(f_v, ipt, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                            # If a node has multiple outputs, gradients is a list of tensors. If it has only a single
                            # output gradients is a tensor. To handle both cases, we first convert gradients to a
                            # list if it's a single tensor.
                            if not isinstance(gradients, list):
                                gradients = [gradients]

                            # Compute the approximation per node's output
                            score_approx_per_output = []
                            for grad in gradients:
                                grad = tf.reshape(grad, [grad.shape[0], -1])
                                score_approx_per_output.append(tf.reduce_mean(tf.reduce_sum(tf.pow(grad, 2.0))))

                            # Free gradients
                            del grad
                            del gradients

                            # If the change to the mean approximation is insignificant (to all outputs)
                            # we stop the calculation.
                            if j > MIN_JACOBIANS_ITER:
                                new_mean_per_output = []
                                delta_per_output = []
                                # Compute new means and deltas for each output index
                                for output_idx, score_approx in enumerate(score_approx_per_output):
                                    prev_scores_output = [x[output_idx] for x in interest_point_scores]
                                    new_mean = np.mean([score_approx, *prev_scores_output])
                                    delta = new_mean - np.mean(prev_scores_output)
                                    new_mean_per_output.append(new_mean)
                                    delta_per_output.append(delta)

                                # Check if all outputs have converged
                                is_converged = all([np.abs(delta) / (np.abs(new_mean) + 1e-6) < JACOBIANS_COMP_TOLERANCE for delta, new_mean in zip(delta_per_output, new_mean_per_output)])
                                if is_converged:
                                    interest_point_scores.append(score_approx_per_output)
                                    break

                            interest_point_scores.append(score_approx_per_output)

                    final_approx_per_output = []
                    # Compute the final approximation for each output index
                    num_node_outputs = len(interest_point_scores[0])
                    for output_idx in range(num_node_outputs):
                        final_approx_per_output.append(2 * tf.reduce_mean([x[output_idx] for x in interest_point_scores]) / output.shape[-1])

                    # final_approx_per_output is a list of all approximations (one per output), thus we average them to
                    # get the final score of a node.
                    trace_approx_by_node.append(tf.reduce_mean(final_approx_per_output))  # Get averaged squared trace approximation

                trace_approx_by_node = tf.reduce_mean([trace_approx_by_node], axis=0)  # Just to get one tensor instead of list of tensors with single element

            # Free gradient tape
            del g

            return trace_approx_by_node.numpy().tolist()

        else:
            Logger.error(f"{self.hessian_request.granularity} is not supported for Keras activation hessian's trace approx calculator")


    def _update_ips_with_outputs_replacements(self,
                                              outputs_replacement_nodes: List[BaseNode],
                                              interest_points: List[BaseNode]):
        """
        Updates the list of interest points with the set of pre-calculated replacement outputs.
        Also, returns the indices of all output nodes (original, replacements and nodes in between them) in a
        topological sorted interest points list (for later use in gradients computation and normalization).

        Returns: A list of indices of the output nodes in the sorted interest points list.

        """

        replacement_outputs_to_ip = [r_node for r_node in outputs_replacement_nodes if
                                     r_node not in interest_points]
        updated_interest_points = interest_points + replacement_outputs_to_ip

        # Re-sort interest points in a topological order according to the graph's sort
        interest_points = [n for n in self.graph.get_topo_sorted_nodes() if n in updated_interest_points]

        output_indices = [interest_points.index(n.node) for n in self.graph.get_outputs()]
        replacement_indices = [interest_points.index(n) for n in outputs_replacement_nodes]
        return list(set(output_indices + replacement_indices))

    def _get_model_output_replacement(self) -> List[str]:
        """
        If a model's output node is not compatible for the task of gradients computation we need to find a predecessor
        node in the model's graph representation which is compatible and use it for the gradients' computation.
        This method searches for this predecessor node for each output of the model.

        Returns: A list of output replacement nodes.

        """

        replacement_outputs = []
        for n in self.graph.get_outputs():
            prev_node = n.node
            while not self.fw_impl.is_node_compatible_for_metric_outputs(prev_node):
                prev_node = self.graph.get_prev_nodes(prev_node)
                assert len(prev_node) == 1, "A none compatible output node has multiple inputs, " \
                                            "which is incompatible for metric computation."
                prev_node = prev_node[0]
            replacement_outputs.append(prev_node)
        return replacement_outputs

    def _get_model_outputs_for_single_image(self,
                                            output_list: List[str],
                                            gradient_tape: tf.GradientTape) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        """
        Computes the model's output according to the given graph representation on the given input,
        while recording necessary intermediate tensors for gradients computation.

        Args:
            output_list: List of nodes that considered as model's output for the purpose of gradients computation.
            gradient_tape: A GradientTape object for recording necessary info for computing gradients.

        Returns: A list of output tensors and a list of activation tensors of all interest points.

        """
        model_input_tensors = {inode: self.fw_impl.to_tensor(self.input_images[i]) for i, inode in
                               enumerate(self.graph.get_inputs())}

        node_to_output_tensors_dict = dict()

        # Build an OperationHandler to handle conversions from graph nodes to Keras operators.
        oh = OperationHandler(self.graph)
        input_nodes_to_input_tensors = {inode: tf.convert_to_tensor(model_input_tensors[inode]) for
                                        inode in self.graph.get_inputs()}  # Cast numpy array to tf.Tensor

        interest_points_tensors = []
        output_tensors = []
        for n in oh.node_sort:
            # Build a dictionary from node to its output tensors, by applying the layers sequentially.
            op_func = oh.get_node_op_function(n)  # Get node operation function

            input_tensors = self._build_input_tensors_list(n,
                                                           self.graph,
                                                           node_to_output_tensors_dict)  # Fetch Node inputs

            out_tensors_of_n = self._run_operation(n,  # Run node operation and fetch outputs
                                                   input_tensors,
                                                   op_func,
                                                   input_nodes_to_input_tensors)

            # Gradients can be computed only on float32 tensors
            if isinstance(out_tensors_of_n, list):
                for i, t in enumerate(out_tensors_of_n):
                    out_tensors_of_n[i] = tf.dtypes.cast(t, tf.float32)
            else:
                out_tensors_of_n = tf.dtypes.cast(out_tensors_of_n, tf.float32)

            if n.name==self.hessian_request.target_node.name:
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

    def _build_input_tensors_list(self,
                                  node: BaseNode,
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

    def _run_operation(self,
                       n: BaseNode,
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
