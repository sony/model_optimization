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
from model_compression_toolkit.core.keras.hessian.trace_hessian_calculator_keras import TraceHessianCalculatorKeras, \
    _concat_outputs
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
                output = _concat_outputs(outputs)

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