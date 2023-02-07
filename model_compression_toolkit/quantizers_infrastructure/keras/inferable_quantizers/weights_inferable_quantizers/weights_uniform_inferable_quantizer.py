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

import numpy as np

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.constants import FOUND_TF
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import mark_quantizer, QuantizationTarget
from model_compression_toolkit.quantizers_infrastructure.common.quant_utils import adjust_range_to_include_zero
from model_compression_toolkit.quantizers_infrastructure.keras.validation_functions import \
    validate_uniform_min_max_ranges, validate_adjusted_min_max_ranges

if FOUND_TF:
    import tensorflow as tf
    from model_compression_toolkit.quantizers_infrastructure.keras.inferable_quantizers.base_keras_inferable_quantizer import BaseKerasInferableQuantizer

    @mark_quantizer(quantization_target=QuantizationTarget.Weights,
                    quantization_method=[QuantizationMethod.UNIFORM],
                    quantizer_type=None)
    class WeightsUniformInferableQuantizer(BaseKerasInferableQuantizer):
        """
        Class for quantizing weights using a uniform quantizer
        """
        def __init__(self,
                     num_bits: int,
                     min_range: List[float],
                     max_range: List[float],
                     per_channel: bool,
                     channel_axis: int = None,
                     input_rank: int = None
                     ):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                min_range: min quantization range for quantizing weights
                max_range: max quantization range for quantizing weights
                per_channel: whether to use per-channel quantization
                channel_axis: axis along which to apply per-channel quantization
                input_rank: number of dimensions of input tensor the quantizer quantizes
            """

            super(WeightsUniformInferableQuantizer, self).__init__()

            # Validate inputs properties
            validate_uniform_min_max_ranges(min_range,
                                            max_range)

            # Convert min/max to numpy arrays
            min_range, max_range = np.asarray(min_range), np.asarray(max_range)
            _min_range, _max_range = adjust_range_to_include_zero(min_range, max_range, num_bits)
            validate_adjusted_min_max_ranges(min_range=min_range,
                                             max_range=max_range,
                                             adj_min=_min_range,
                                             adj_max=_max_range)

            self.num_bits = num_bits
            self.max_range = _max_range
            self.min_range = _min_range

            if per_channel:
                assert input_rank is not None, f'Input rank is missing in per channel quantization'
                assert channel_axis is not None, f'Channel axis is missing in per channel quantization'
                assert len(self.min_range) >= 1, f'In per-channel quantization min ranges list should be of length >= 1 but is {len(self.min_range)}'
                assert len(self.max_range) >= 1, f'In per-channel quantization max ranges list should be of length >= 1 but is {len(self.max_range)}'
            else:
                assert len(self.min_range) == 1, f'In per-tensor quantization min/max should be of length 1 but is {len(min_range)}'
                assert len(self.min_range) == 1, f'In per-tensor quantization min_range should be of length 1 but is {len(self.min_range)}'
                assert len(self.max_range) == 1, f'In per-tensor quantization max_range should be of length 1 but is {len(self.max_range)}'
                self.min_range = self.min_range[0]
                self.max_range = self.max_range[0]

            self.per_channel = per_channel
            self.channel_axis = channel_axis
            self.input_rank = input_rank

            # Tensorflow's fake_quant_with_min_max_vars_per_channel only works on last axis, so
            # need to move the quantization axis to the last axis
            if per_channel and channel_axis not in [-1, self.input_rank - 1]:
                # If per-channel quantization is being used and the channel axis is not the last axis,
                # create a permutation vector to move the channel axis to the last position
                self.perm_vec = list(np.arange(self.input_rank))
                self.perm_vec[channel_axis] = self.input_rank - 1
                self.perm_vec[self.input_rank - 1] = channel_axis
            else:
                # If per-channel quantization is not being used or the channel axis is already the last axis,
                # set the permutation vector to None
                self.perm_vec = None

        def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize

            Returns:
                quantized tensor.
            """
            assert inputs.dtype==tf.float32, f'Input tensor was expected to be a float tensor but is of type {inputs.dtype}'

            # If per-channel quantization is being used
            if self.per_channel:
                # If a permutation vector has been created to move the channel axis to the last position
                if self.perm_vec:
                    # Transpose the input tensor to move the channel axis to the last position
                    inputs = tf.transpose(inputs, perm=self.perm_vec)

                # Quantize the input tensor using per-channel quantization
                q_tensor = tf.quantization.fake_quant_with_min_max_vars_per_channel(inputs,
                                                                                    min=self.min_range,
                                                                                    max=self.max_range,
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
                Dictionary with the following keys: 'num_bits', 'min_range', 'max_range', 'per_channel', 'channel_axis'
            """
            return {'per_channel': self.per_channel,
                    'num_bits': self.num_bits,
                    'max_range': self.max_range,
                    'min_range': self.min_range,
                    'channel_axis': self.channel_axis,
                    'input_rank': self.input_rank}


else:
    class WeightsUniformInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing tensorflow and tensorflow_model_optimization is mandatory '
                            'when using WeightsUniformInferableQuantizer. '
                            'Could not find Tensorflow package.')
