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
from model_compression_toolkit.core.common.hessian import HessianScoresRequest, HessianScoresGranularity
from model_compression_toolkit.core.keras.back2framework.float_model_builder import FloatKerasModelBuilder
from model_compression_toolkit.core.keras.hessian.hessian_scores_calculator_keras import HessianScoresCalculatorKeras
from model_compression_toolkit.logger import Logger


class ActivationHessianScoresCalculatorKeras(HessianScoresCalculatorKeras):
    """
    Keras implementation of the Hessian-approximation scores Calculator for activations.
    """
    def __init__(self,
                 graph: Graph,
                 input_images: List[tf.Tensor],
                 fw_impl,
                 hessian_scores_request: HessianScoresRequest,
                 num_iterations_for_approximation: int = HESSIAN_NUM_ITERATIONS):
        """
        Args:
            graph: Computational graph for the float model.
            input_images: List of input images for the computation.
            fw_impl: Framework-specific implementation for Hessian approximation scores computation.
            hessian_scores_request: Configuration request for which to compute the Hessian approximation scores.
            num_iterations_for_approximation: Number of iterations to use when approximating the Hessian scores.

        """
        super(ActivationHessianScoresCalculatorKeras, self).__init__(graph=graph,
                                                                     input_images=input_images,
                                                                     fw_impl=fw_impl,
                                                                     hessian_scores_request=hessian_scores_request,
                                                                     num_iterations_for_approximation=num_iterations_for_approximation)

    def compute(self) -> List[np.ndarray]:
        """
        Compute the Hessian-approximation based scores w.r.t the requested target nodes' activations.

        Returns:
            List[np.ndarray]: Scores based on the Hessian-approximation for the requested nodes.
        """
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

            # List to store the Hessian-approximation scores for each interest point
            ipts_hessian_approximations = [tf.Variable([0.0], dtype=tf.float32, trainable=True)
                                           for _ in range(len(target_activation_tensors))]

            # Loop through each interest point activation tensor
            prev_mean_results = None
            for j in tqdm(range(self.num_iterations_for_approximation)):  # Approximation iterations
                # Generate random tensor of 1s and -1s
                v = self._generate_random_vectors_batch(output.shape)
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

                        if self.hessian_request.granularity == HessianScoresGranularity.PER_TENSOR:
                            # Mean over all dims but the batch (CXHXW for conv)
                            hessian_approx = tf.reduce_sum(hess_v ** 2.0,
                                                           axis=tuple(d for d in range(1, len(hess_v.shape))))
                        elif self.hessian_request.granularity == HessianScoresGranularity.PER_ELEMENT:
                            hessian_approx = hess_v ** 2
                        elif self.hessian_request.granularity == HessianScoresGranularity.PER_OUTPUT_CHANNEL:
                            axes_to_sum = tuple(d for d in range(1, len(hess_v.shape)-1))
                            hessian_approx = tf.reduce_sum(hess_v ** 2.0, axis=axes_to_sum)

                        else:  # pragma: no cover
                            Logger.critical(f"{self.hessian_request.granularity} "
                                            f"is not supported for Keras activation hessian\'s approximation scores calculator.")

                        # Free gradients
                        del hess_v

                        # Update node Hessian approximation mean over random iterations
                        ipts_hessian_approximations[i] = (j * ipts_hessian_approximations[i] + hessian_approx) / (j + 1)

                # If the change to the mean approximation is insignificant (to all outputs)
                # we stop the calculation.
                if j > MIN_HESSIAN_ITER and prev_mean_results is not None:
                    new_mean_res = tf.reduce_mean(tf.stack(ipts_hessian_approximations), axis=1)
                    relative_delta_per_node = (tf.abs(new_mean_res - prev_mean_results) /
                                               (tf.abs(new_mean_res) + 1e-6))
                    max_delta = tf.reduce_max(relative_delta_per_node)
                    if max_delta < HESSIAN_COMP_TOLERANCE:
                        break

                if self.hessian_request.granularity == HessianScoresGranularity.PER_TENSOR:
                    prev_mean_results = tf.reduce_mean(tf.stack(ipts_hessian_approximations), axis=1)

            # Convert results to list of numpy arrays
            hessian_results = [h.numpy() for h in ipts_hessian_approximations]
            # Extend the Hessian tensors shape to align with expected return type
            # TODO: currently, only per-tensor Hessian is available for activation.
            #  Once implementing per-channel or per-element, this alignment needs to be verified and handled separately.
            hessian_results = [h[..., np.newaxis] for h in hessian_results]

            return hessian_results

