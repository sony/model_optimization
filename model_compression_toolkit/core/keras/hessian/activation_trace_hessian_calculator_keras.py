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

from typing import List

import tensorflow as tf
from tqdm import tqdm
import numpy as np

from model_compression_toolkit.constants import MIN_HESSIAN_ITER, HESSIAN_COMP_TOLERANCE, \
    HESSIAN_NUM_ITERATIONS
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.hessian import TraceHessianRequest, HessianInfoGranularity
from model_compression_toolkit.core.keras.back2framework.float_model_builder import FloatKerasModelBuilder
from model_compression_toolkit.core.keras.hessian.trace_hessian_calculator_keras import TraceHessianCalculatorKeras
from model_compression_toolkit.logger import Logger


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
            model_output_nodes = [ot.node for ot in self.graph.get_outputs()]

            if self.hessian_request.target_node in model_output_nodes:
                Logger.critical("Trying to compute activation Hessian approximation with respect to the model output. "
                                 "This operation is not supported. "
                                 "Remove the output node from the set of node targets in the Hessian request.")

            grad_model_outputs = [self.hessian_request.target_node] + model_output_nodes

            # Building a model to run Hessian approximation on
            model, _ = FloatKerasModelBuilder(graph=self.graph, append2output=grad_model_outputs).build_model()

            # Record operations for automatic differentiation
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
                g.watch(self.input_images)

                if len(self.input_images) > 1:
                    outputs = model(self.input_images)
                else:
                    outputs = model(*self.input_images)

                if len(outputs) != len(grad_model_outputs):
                    Logger.critical(
                        f"Model for computing activation Hessian approximation expects {len(grad_model_outputs)} "
                        f"outputs, but got {len(outputs)} output tensors.")

                # Extracting the intermediate activation tensors and the model real output
                # TODO: we assume that the hessian request is for a single node.
                #  When we extend it to multiple nodes in the same request, then we should modify this part to take
                #  the first "num_target_nodes" outputs from the output list.
                #  We also assume that the target nodes are not part of the model output nodes, if this assumption changed,
                #  then the code should be modified accordingly.
                target_activation_tensors = [outputs[0]]
                output_tensors = outputs[1:]

                # Unfold and concatenate all outputs to form a single tensor
                output = self._concat_tensors(output_tensors)

                # List to store the approximated trace of the Hessian for each interest point
                trace_approx_by_node = []
                # Loop through each interest point activation tensor
                for ipt in tqdm(target_activation_tensors):  # Per Interest point activation tensor
                    interest_point_scores = []  # List to store scores for each interest point
                    for j in range(self.num_iterations_for_approximation):  # Approximation iterations
                        # Getting a random vector with normal distribution
                        v = tf.random.normal(shape=output.shape, dtype=output.dtype)
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
                                score_approx_per_output.append(tf.reduce_sum(tf.pow(grad, 2.0)))

                            # Free gradients
                            del grad
                            del gradients

                            # If the change to the mean approximation is insignificant (to all outputs)
                            # we stop the calculation.
                            if j > MIN_HESSIAN_ITER:
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
                                is_converged = all([np.abs(delta) / (np.abs(new_mean) + 1e-6) < HESSIAN_COMP_TOLERANCE for delta, new_mean in zip(delta_per_output, new_mean_per_output)])
                                if is_converged:
                                    interest_point_scores.append(score_approx_per_output)
                                    break

                            interest_point_scores.append(score_approx_per_output)

                    final_approx_per_output = []
                    # Compute the final approximation for each output index
                    num_node_outputs = len(interest_point_scores[0])
                    for output_idx in range(num_node_outputs):
                        final_approx_per_output.append(tf.reduce_mean([x[output_idx] for x in interest_point_scores]))

                    # final_approx_per_output is a list of all approximations (one per output), thus we average them to
                    # get the final score of a node.
                    trace_approx_by_node.append(tf.reduce_mean(final_approx_per_output))  # Get averaged squared trace approximation

                trace_approx_by_node = tf.reduce_mean([trace_approx_by_node], axis=0)  # Just to get one tensor instead of list of tensors with single element

            # Free gradient tape
            del g

            return trace_approx_by_node.numpy().tolist()

        else:
            Logger.critical(f"{self.hessian_request.granularity} is not supported for Keras activation hessian\'s trace approximation calculator.")
