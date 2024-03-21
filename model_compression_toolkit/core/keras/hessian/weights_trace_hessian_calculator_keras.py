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

import numpy as np
import tensorflow as tf
from typing import List

from model_compression_toolkit.constants import HESSIAN_NUM_ITERATIONS, MIN_HESSIAN_ITER, HESSIAN_COMP_TOLERANCE, HESSIAN_EPS
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.hessian import TraceHessianRequest, HessianInfoGranularity
from model_compression_toolkit.core.keras.back2framework.float_model_builder import FloatKerasModelBuilder
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.hessian.trace_hessian_calculator_keras import TraceHessianCalculatorKeras
from model_compression_toolkit.logger import Logger


class WeightsTraceHessianCalculatorKeras(TraceHessianCalculatorKeras):
    """
    Keras-specific implementation of the Trace Hessian approximation computation w.r.t a node's weights.
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
            fw_impl: Framework-specific implementation for trace Hessian computation.
            trace_hessian_request: Configuration request for which to compute the trace Hessian approximation.
            num_iterations_for_approximation: Number of iterations to use when approximating the Hessian trace.
        """
        super(WeightsTraceHessianCalculatorKeras, self).__init__(graph=graph,
                                                                 input_images=input_images,
                                                                 fw_impl=fw_impl,
                                                                 trace_hessian_request=trace_hessian_request,
                                                                 num_iterations_for_approximation=num_iterations_for_approximation)

    def compute(self) -> np.ndarray:
        """
        Compute the Hessian-based scores w.r.t target node's weights.
        Currently, supported nodes are [Conv2D, Dense, Conv2DTranspose, DepthwiseConv2D].
        The computed scores are returned in a numpy array. The shape of the result differs
        according to the requested granularity. If for example the node is Conv2D with a kernel
        shape of (3, 3, 3, 2) (namely, 3 input channels, 2 output channels and kernel size of 3x3)
        and the required granularity is HessianInfoGranularity.PER_TENSOR the result shape will be (1,),
        for HessianInfoGranularity.PER_OUTPUT_CHANNEL the shape will be (2,) and for
        HessianInfoGranularity.PER_ELEMENT a shape of (3, 3, 3, 2).

        Returns:  The computed scores as numpy ndarray.

        """
        # Check if the target node's layer type is supported
        if not DEFAULT_KERAS_INFO.is_kernel_op(self.hessian_request.target_node.type):
            Logger.critical(
                f"{self.hessian_request.target_node.type} is not supported for Hessian-based scoring with respect to weights.")

        # Construct the Keras float model for inference
        model, _ = FloatKerasModelBuilder(graph=self.graph).build_model()

        # Get the weight attributes for the target node type
        weight_attributes = DEFAULT_KERAS_INFO.get_kernel_op_attributes(self.hessian_request.target_node.type)

        # Get the weight tensor for the target node
        if len(weight_attributes) != 1:
            Logger.critical(f"Hessian-based scoring with respect to weights is currently supported only for nodes with a single weight attribute. Found {len(weight_attributes)} attributes.")

        weight_tensor = getattr(model.get_layer(self.hessian_request.target_node.name), weight_attributes[0])

        # Get the output channel index (needed for HessianInfoGranularity.PER_OUTPUT_CHANNEL case)
        output_channel_axis, _ = DEFAULT_KERAS_INFO.kernel_channels_mapping.get(self.hessian_request.target_node.type)

        # Get number of scores that should be calculated by the granularity.
        num_of_scores = self._get_num_scores_by_granularity(weight_tensor,
                                                            output_channel_axis)

        # Initiate a gradient tape for automatic differentiation
        with tf.GradientTape(persistent=True) as tape:
            # Perform a forward pass (inference) to get the output, while watching
            # the input tensor for gradient computation
            tape.watch(self.input_images)
            outputs = model(self.input_images)

            # Combine outputs if the model returns multiple output tensors
            output = self._concat_tensors(outputs)

            approximation_per_iteration = []
            for j in range(self.num_iterations_for_approximation):  # Approximation iterations
                # Getting a random vector with normal distribution and the same shape as the model output
                v = tf.random.normal(shape=output.shape)
                f_v = tf.reduce_sum(v * output)

                # Stop recording operations for automatic differentiation
                with tape.stop_recording():
                    # Compute gradients of f_v with respect to the weights
                    gradients = tape.gradient(f_v, weight_tensor)
                    gradients = self._reshape_gradients(gradients,
                                                        output_channel_axis,
                                                        num_of_scores)
                    approx = tf.reduce_sum(tf.pow(gradients, 2.0), axis=1)

                    # Free gradients
                    del gradients

                    # If the change to the mean approximation is insignificant (to all outputs)
                    # we stop the calculation.
                    if j > MIN_HESSIAN_ITER:
                        # Compute new means and deltas
                        new_mean = tf.reduce_mean(tf.stack(approximation_per_iteration + approx), axis=0)
                        delta = new_mean - tf.reduce_mean(tf.stack(approximation_per_iteration), axis=0)
                        is_converged = np.all(np.abs(delta) / (np.abs(new_mean) + HESSIAN_EPS) < HESSIAN_COMP_TOLERANCE)
                        if is_converged:
                            approximation_per_iteration.append(approx)
                            break

                    approximation_per_iteration.append(approx)

            # Compute the mean of the approximations
            final_approx = tf.reduce_mean(tf.stack(approximation_per_iteration), axis=0)

        # Free gradient tape
        del tape

        if self.hessian_request.granularity == HessianInfoGranularity.PER_TENSOR:
            if final_approx.shape != (1,):
                Logger.critical(f"For HessianInfoGranularity.PER_TENSOR, the expected score shape is (1,), but found {final_approx.shape}.")
        elif self.hessian_request.granularity == HessianInfoGranularity.PER_ELEMENT:
            # Reshaping the scores to the original weight shape
            final_approx = tf.reshape(final_approx, weight_tensor.shape)

        return final_approx.numpy()

    def _reshape_gradients(self,
                           gradients: tf.Tensor,
                           output_channel_axis: int,
                           num_of_scores: int) -> tf.Tensor:
        """
        Reshape the gradient tensor based on the requested granularity.

        The gradients can be reshaped to represent the whole tensor, on a per-output-channel basis,
        or on a per-element basis.

        Args:
            gradients (tf.Tensor): The gradient tensor to be reshaped.
            output_channel_axis (int): The axis in the gradient tensor representing output channels.

        Returns:
            tf.Tensor: Reshaped gradient tensor based on the granularity.
        """
        # Reshape the gradients based on the granularity (whole tensor, per channel, or per element)
        if self.hessian_request.granularity != HessianInfoGranularity.PER_OUTPUT_CHANNEL:
            gradients = tf.reshape(gradients, [num_of_scores, -1])
        else:
            # Slice the gradients, vectorize them and stack them along the first axis.
            # If for example we have n output-channels, and each one of the filters have m
            # elements, the gradients will have the shape of (n,m).
            slices = tf.split(gradients, num_or_size_splits=num_of_scores, axis=output_channel_axis)
            gradients = tf.stack([tf.reshape(slice, (-1,)) for slice in slices])
        return gradients

    def _get_num_scores_by_granularity(self,
                                       weight_tensor: tf.Tensor,
                                       output_channel_axis: int) -> int:
        """
        Get the number of scores to be computed based on the granularity type.

        Args:
            weight_tensor (tf.Tensor): The weight tensor.
            output_channel_axis (int): Axis corresponding to the output channels.

        Returns:
            int: The number of scores.
        """
        if self.hessian_request.granularity == HessianInfoGranularity.PER_TENSOR:
            return 1
        elif self.hessian_request.granularity == HessianInfoGranularity.PER_OUTPUT_CHANNEL:
            return weight_tensor.shape[output_channel_axis]
        elif self.hessian_request.granularity == HessianInfoGranularity.PER_ELEMENT:
            return tf.size(weight_tensor).numpy()
        else:
            Logger.critical(f"Unexpected granularity encountered: {self.hessian_request.granularity}.")
