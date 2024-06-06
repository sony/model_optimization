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

    def compute(self) -> List[np.ndarray]:
        """
        Compute the approximation of the trace of the Hessian w.r.t the requested target nodes' activations.

        Returns:
            List[np.ndarray]: Approximated trace of the Hessian for the requested nodes.
        """
        if self.hessian_request.granularity == HessianInfoGranularity.PER_TENSOR:
            model_output_nodes = [ot.node for ot in self.graph.get_outputs()]

            if len([n for n in self.hessian_request.target_nodes if n in model_output_nodes]) > 0:
                Logger.critical("Trying to compute activation Hessian approximation with respect to the model output. "
                                "This operation is not supported. "
                                "Remove the output node from the set of node targets in the Hessian request.")

            grad_model_outputs = self.hessian_request.target_nodes + model_output_nodes

            # Building a model to run Hessian approximation on
            model, _ = FloatKerasModelBuilder(graph=self.graph, append2output=grad_model_outputs).build_model()

            # Record operations for automatic differentiation
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
                g.watch(self.input_images)

                if len(self.input_images) > 1:
                    outputs = model(self.input_images)
                else:
                    outputs = model(*self.input_images)

                if len(outputs) != len(grad_model_outputs):  # pragma: no cover
                    Logger.critical(
                        f"Model for computing activation Hessian approximation expects {len(grad_model_outputs)} "
                        f"outputs, but got {len(outputs)} output tensors.")

                # Extracting the intermediate activation tensors and the model real output.
                # Note that we do not allow computing Hessian for output nodes, so there shouldn't be an overlap.
                num_target_nodes = len(self.hessian_request.target_nodes)
                # Extract activation tensors of nodes for which we want to compute Hessian
                target_activation_tensors = outputs[:num_target_nodes]
                # Extract the model outputs
                output_tensors = outputs[num_target_nodes:]

                # Unfold and concatenate all outputs to form a single tensor
                output = self._concat_tensors(output_tensors)

                # List to store the approximated trace of the Hessian for each interest point
                ipts_hessian_trace_approx = [tf.Variable([0.0], dtype=tf.float32, trainable=True)
                                             for _ in range(len(target_activation_tensors))]

                # Loop through each interest point activation tensor
                prev_mean_results = None
                for j in tqdm(range(self.num_iterations_for_approximation)):  # Approximation iterations
                    # Getting a random vector with normal distribution
                    v = tf.random.normal(shape=output.shape, dtype=output.dtype)
                    f_v = tf.reduce_sum(v * output)
                    for i, ipt in enumerate(target_activation_tensors):  # Per Interest point activation tensor
                        interest_point_scores = []  # List to store scores for each interest point
                        with g.stop_recording():
                            # Computing the approximation by getting the gradient of (output * v)
                            hess_v = g.gradient(f_v, ipt)

                            if hess_v is None:
                                # In case we have an output node, which is an interest point, but it is not
                                # differentiable, we consider its Hessian to be the initial value 0.
                                continue  # pragma: no cover

                            # Mean over all dims but the batch (CXHXW for conv)
                            hessian_trace_approx = tf.reduce_sum(hess_v ** 2.0,
                                                                 axis=tuple(d for d in range(1, len(hess_v.shape))))

                            # Free gradients
                            del hess_v

                            # Update node Hessian approximation mean over random iterations
                            ipts_hessian_trace_approx[i] = (j * ipts_hessian_trace_approx[i] + hessian_trace_approx) / (j + 1)

                    # If the change to the mean approximation is insignificant (to all outputs)
                    # we stop the calculation.
                    if j > MIN_HESSIAN_ITER:
                        if prev_mean_results is not None:
                            new_mean_res = tf.reduce_mean(tf.stack(ipts_hessian_trace_approx), axis=1)
                            relative_delta_per_node = (tf.abs(new_mean_res - prev_mean_results) /
                                                       (tf.abs(new_mean_res) + 1e-6))
                            max_delta = tf.reduce_max(relative_delta_per_node)
                            if max_delta < HESSIAN_COMP_TOLERANCE:
                                break
                    prev_mean_results = tf.reduce_mean(tf.stack(ipts_hessian_trace_approx), axis=1)

                # Convert results to list of numpy arrays
                hessian_results = [h.numpy() for h in ipts_hessian_trace_approx]
                # Extend the Hessian tensors shape to align with expected return type
                # TODO: currently, only per-tensor Hessian is available for activation.
                #  Once implementing per-channel or per-element, this alignment needs to be verified and handled separately.
                hessian_results = [h[..., np.newaxis] for h in hessian_results]

                return hessian_results

        else:  # pragma: no cover
            Logger.critical(f"{self.hessian_request.granularity} "
                            f"is not supported for Keras activation hessian\'s trace approximation calculator.")
