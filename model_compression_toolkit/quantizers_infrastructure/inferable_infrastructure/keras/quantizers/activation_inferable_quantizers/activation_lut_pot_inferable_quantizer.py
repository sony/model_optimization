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

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.constants import FOUND_TF

from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from model_compression_toolkit.quantizers_infrastructure import QuantizationTarget
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.common.base_inferable_quantizer import mark_quantizer
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.common.constants import MULTIPLIER_N_BITS, EPS

if FOUND_TF:
    import tensorflow as tf
    from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.keras.quantizers.base_keras_inferable_quantizer \
        import BaseKerasInferableQuantizer
    from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.keras.quantizer_utils import lut_quantizer


    @mark_quantizer(quantization_target=QuantizationTarget.Activation,
                    quantization_method=[QuantizationMethod.LUT_POT_QUANTIZER],
                    quantizer_type=None)
    class ActivationLutPOTInferableQuantizer(BaseKerasInferableQuantizer):
        """
        Class for quantizing activations using lut power-of-two quantizer
        """

        def __init__(self,
                     num_bits: int,
                     cluster_centers: np.ndarray,
                     threshold: List[float],
                     signed: bool,
                     multiplier_n_bits: int = MULTIPLIER_N_BITS,
                     eps: float = EPS):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                cluster_centers: the cluster centers to assign the activations
                threshold: threshold for quantizing activations
                signed: whether or not to use signed quantization
                multiplier_n_bits: Number of bits that determines the quantization range
                eps: Small value for numerical stability in division
            """
            # Call the superclass constructor with the given parameters, along with the target of Activation
            # quantization
            super(ActivationLutPOTInferableQuantizer, self).__init__()

            assert isinstance(threshold, list), f'Expected threshold to be of type list but is {type(threshold)}'
            assert all([isinstance(x, (float, np.float32, tf.float32)) for x in
                        threshold]), f'Expected threshold list to contain float or np.float values but found ' \
                                     f'{[type(x) for x in threshold]}'

            # In activation per-channel quantization is not supported thus we expect a single threshold value.
            assert len(threshold) == 1, f'In per-tensor quantization threshold should be of ' \
                                        f'length 1 but is {len(threshold)}'

            is_threshold_pot = np.all([int(np.log2(x)) == np.log2(x) for x in threshold])
            assert is_threshold_pot, f'Expected threshold to be power of 2 but is {threshold}'

            self.threshold = threshold[0]

            assert len(np.unique(cluster_centers)) <= 2 ** num_bits, \
                f'Expected num of cluster centers to be less or equal than {2 ** num_bits} ' \
                f'but got {len(cluster_centers)}'

            assert not np.any(cluster_centers - cluster_centers.astype(int)), f'Expected cluster centers to be integers'

            if signed:
                assert np.all((-1 * (2 ** (multiplier_n_bits - int(signed))) <= cluster_centers) &
                              (cluster_centers <= (2 ** (multiplier_n_bits - int(signed)) - 1))), \
                    f'Expected cluster centers in the quantization range'
            else:
                assert np.all(cluster_centers <= (2 ** multiplier_n_bits)), \
                    f'Expected cluster centers in the quantization range'

            # num_bits must be less than multiplier_n_bits
            assert num_bits <= multiplier_n_bits, f'Look-Up-Table bit configuration has {num_bits} bits. It must be ' \
                                                  f'less then {multiplier_n_bits}'
            if num_bits == multiplier_n_bits:
                warnings.warn("Num of bits equal to multiplier n bits, Please be aware LUT quantizier may be "
                              "inefficient in that case, consider using SymmetricInferableQuantizer instead")

            # If unsigned activation quantization, all cluster_centers must have the same sign
            if not signed:
                assert np.all(cluster_centers >= 0), f'Expected unsigned cluster centers in unsigned activation ' \
                                                     f'quantization '

            self.num_bits = num_bits
            self.cluster_centers = cluster_centers
            self.signed = signed
            self.multiplier_n_bits = multiplier_n_bits
            self.eps = eps

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

            return lut_quantizer(inputs, cluster_centers=self.cluster_centers, signed=self.signed,
                                 threshold=self.threshold, multiplier_n_bits=self.multiplier_n_bits, eps=self.eps)

        def get_config(self):
            """
            Return a dictionary with the configuration of the quantizer.

            Returns:
                Dictionary with the following keys: 'num_bits', 'cluster_centers', 'threshold', 'signed',
                'multiplier_n_bits', 'eps'
            """
            return {'num_bits': self.num_bits,
                    'cluster_centers': self.cluster_centers,
                    'threshold': self.threshold,
                    'signed': self.signed,
                    'multiplier_n_bits': self.multiplier_n_bits,
                    'eps': self.eps}


else:
    class ActivationLutPOTInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            Logger.error('Installing tensorflow and tensorflow_model_optimization is mandatory '
                         'when using ActivationLutPOTInferableQuantizer. '
                         'Could not find Tensorflow package.')
