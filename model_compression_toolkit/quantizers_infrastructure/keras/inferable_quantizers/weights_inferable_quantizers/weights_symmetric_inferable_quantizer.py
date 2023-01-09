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

from model_compression_toolkit.core.common.constants import FOUND_TF
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import QuantizationTarget

if FOUND_TF:
    import tensorflow as tf
    from model_compression_toolkit.quantizers_infrastructure.keras.inferable_quantizers\
        .base_symmetric_inferable_quantizer import \
        BaseSymmetricInferableQuantizer

    class WeightsSymmetricInferableQuantizer(BaseSymmetricInferableQuantizer):
        """
        Class for quantizing weights using a symmetric quantizer
        """
        def __init__(self,
                     num_bits: int,
                     threshold: np.ndarray,
                     signed: bool,
                     per_channel: bool,
                     channel_axis: int):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                threshold: threshold for quantizing weights
                signed: whether or not to use signed quantization
                per_channel: whether to use per-channel quantization
                channel_axis: axis along which to apply per-channel quantization
            """

            super(WeightsSymmetricInferableQuantizer, self).__init__(num_bits=num_bits,
                                                                     threshold=threshold,
                                                                     signed=signed,
                                                                     quantization_target=QuantizationTarget.Weights)

            self.per_channel = per_channel
            self.channel_axis = channel_axis

            # Get the shape of the threshold array
            self.threshold_shape = np.asarray(threshold).shape

            # Tensorflow's fake_quant_with_min_max_vars_per_channel only works on last axis, so
            # need to move the quantization axis to the last axis
            if per_channel and channel_axis not in [-1, len(self.threshold_shape) - 1]:
                # If per-channel quantization is being used and the channel axis is not the last axis,
                # create a permutation vector to move the channel axis to the last position
                self.perm_vec = list(np.arange(len(self.threshold_shape)))
                self.perm_vec[channel_axis] = len(self.threshold_shape) - 1
                self.perm_vec[len(self.threshold_shape) - 1] = channel_axis
            else:
                # If per-channel quantization is not being used or the channel axis is already the last axis,
                # set the permutation vector to None
                self.perm_vec = None

        def __call__(self, inputs, training=False):
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize
                training: whether or not the quantizer is being used in training mode (unused here)

            Returns:
                quantized tensor.
            """
            # If per-channel quantization is being used
            if self.per_channel:
                # If a permutation vector has been created to move the channel axis to the last position
                if self.perm_vec:
                    # Transpose the input tensor to move the channel axis to the last position
                    inputs = tf.transpose(inputs, perm=self.perm_vec)

                # Quantize the input tensor using per-channel quantization
                q_tensor = tf.quantization.fake_quant_with_min_max_vars_per_channel(inputs,
                                                                                    min=self.min_range.flatten(),
                                                                                    max=self.max_range.flatten(),
                                                                                    num_bits=self.num_bits)
                if self.perm_vec:
                    # Transpose the quantized tensor back to its original shape
                    q_tensor = tf.transpose(q_tensor, perm=self.perm_vec)

                # Return the quantized tensor
                return q_tensor
            else:
                # If per-channel quantization is not being used, quantize the input tensor using regular quantization
                return tf.quantization.fake_quant_with_min_max_vars(inputs,
                                                                    min=self.min_range,
                                                                    max=self.max_range,
                                                                    num_bits=self.num_bits)
        def get_config(self):
            """
            Return a dictionary with the configuration of the quantizer.

            Returns:
                Dictionary with the following keys: 'num_bits', 'signed', 'threshold', 'per_channel', 'channel_axis'
            """
            return {'num_bits': self.num_bits,
                    'signed': self.signed,
                    'threshold': self.threshold,
                    'per_channel': self.per_channel,
                    'channel_axis': self.channel_axis}

else:
    class WeightsSymmetricInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing tensorflow and tensorflow_model_optimization is mandatory '
                            'when using WeightsPOTInferableQuantizer. '
                            'Could not find Tensorflow package.')
