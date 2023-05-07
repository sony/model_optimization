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
import warnings
from typing import List

import numpy as np

from model_compression_toolkit.constants import FOUND_TF
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.common.base_inferable_quantizer import mark_quantizer, \
    QuantizationTarget
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.common.constants import MULTIPLIER_N_BITS, EPS

if FOUND_TF:
    import tensorflow as tf
    from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.keras.quantizers.base_keras_inferable_quantizer import \
        BaseKerasInferableQuantizer
    from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.keras.quantizer_utils import \
        lut_quantizer

    @mark_quantizer(quantization_target=QuantizationTarget.Weights,
                    quantization_method=[QuantizationMethod.LUT_SYM_QUANTIZER],
                    quantizer_type=None)
    class WeightsLUTSymmetricInferableQuantizer(BaseKerasInferableQuantizer):
        """
        Class for quantizing weights using a lut symmetric quantizer
        """

        def __init__(self,
                     num_bits: int,
                     cluster_centers: np.ndarray,
                     threshold: List[float],
                     per_channel: bool,
                     channel_axis: int = None,
                     input_rank: int = None,
                     multiplier_n_bits: int = MULTIPLIER_N_BITS,
                     eps: float = EPS):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                cluster_centers: the cluster centers to assign the weights
                threshold: threshold for quantizing weights
                per_channel: whether to use per-channel quantization
                channel_axis: axis along which to apply per-channel quantization
                input_rank: number of dimensions of input tensor the quantizer quantizes
                multiplier_n_bits: Number of bits that determines the quantization range
                eps: Small value for numerical stability in division
            """

            super(WeightsLUTSymmetricInferableQuantizer, self).__init__()

            assert isinstance(threshold, list), f'Expected threshold to be of type list but is {type(threshold)}'
            assert all([isinstance(x, (float, np.float32, np.float64)) for x in
                        threshold]), f'Expected threshold list to contain float or np.float values but found ' \
                                     f'{[type(x) for x in threshold]}'

            self.threshold = np.asarray(threshold)

            if per_channel:
                assert input_rank is not None, f'Input rank is missing in per channel quantization'
                assert channel_axis is not None, f'Channel axis is missing in per channel quantization'
                assert len(threshold) >= 1, f'In per-channel quantization threshold list should be of length >= 1 ' \
                                            f'but is {len(threshold)} '
            else:
                assert len(threshold) == 1, f'In per-tensor quantization threshold should be of length 1 but is' \
                                            f' {len(threshold)}'
                self.threshold = self.threshold[0]

            assert len(np.unique(cluster_centers)) <= 2 ** num_bits, \
                f'Expected num of cluster centers to be less or equal than {2 ** num_bits} ' \
                f'but got {len(cluster_centers)}'

            assert not np.any(cluster_centers - cluster_centers.astype(int)), f'Expected cluster centers to be integers'

            # Weight quantization is signed, hence the quantization range is
            # [-2**(multiplier_n_bits - 1), 2**(multiplier_n_bits - 1) - 1]
            assert np.all((-1 * (2 ** (multiplier_n_bits - 1)) <= cluster_centers) &
                          (cluster_centers <= (2 ** (multiplier_n_bits - 1) - 1))), \
                f'Expected cluster centers in the quantization range'

            # num_bits must be less than multiplier_n_bits
            assert num_bits <= multiplier_n_bits, f'Look-Up-Table bit configuration has {num_bits} bits. It must be ' \
                                                  f'less then {multiplier_n_bits}'
            if num_bits == multiplier_n_bits:
                warnings.warn("Num of bits equal to multiplier n bits, Please be aware LUT quantizier may be "
                              "inefficient in that case, consider using SymmetricInferableQuantizer instead")

            self.num_bits = num_bits
            self.cluster_centers = cluster_centers
            self.multiplier_n_bits = multiplier_n_bits
            self.eps = eps
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
            assert inputs.dtype == tf.float32, f'Input tensor was expected to be a float tensor but is of type ' \
                                               f'{inputs.dtype}'

            # If per-channel quantization is being used
            if self.per_channel:
                # If a permutation vector has been created to move the channel axis to the last position
                if self.perm_vec:
                    # Transpose the input tensor to move the channel axis to the last position
                    inputs = tf.transpose(inputs, perm=self.perm_vec)

                # Quantize the input tensor using per-channel quantization
                q_tensor = lut_quantizer(inputs, cluster_centers=self.cluster_centers, signed=True,
                                         threshold=self.threshold, multiplier_n_bits=self.multiplier_n_bits,
                                         eps=self.eps)
                if self.perm_vec:
                    # Transpose the quantized tensor back to its original shape
                    q_tensor = tf.transpose(q_tensor, perm=self.perm_vec)

                # Return the quantized tensor
                return q_tensor
            else:
                return lut_quantizer(inputs, cluster_centers=self.cluster_centers, signed=True,
                                     threshold=self.threshold, multiplier_n_bits=self.multiplier_n_bits, eps=self.eps)

        def get_config(self):
            """
            Return a dictionary with the configuration of the quantizer.

            Returns:
                Dictionary with the following keys: 'per_channel', 'num_bits', 'cluster_centers', 'threshold',
                 'channel_axis', 'input_rank', 'multiplier_n_bits', 'eps'
            """
            return {'per_channel': self.per_channel,
                    'num_bits': self.num_bits,
                    'cluster_centers': self.cluster_centers,
                    'threshold': self.threshold,
                    'channel_axis': self.channel_axis,
                    'input_rank': self.input_rank,
                    'multiplier_n_bits': self.multiplier_n_bits,
                    'eps': self.eps}

else:
    class WeightsLUTSymmetricInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing tensorflow and tensorflow_model_optimization is mandatory '
                            'when using WeightsLUTSymmetricInferableQuantizer. '
                            'Could not find Tensorflow package.')
