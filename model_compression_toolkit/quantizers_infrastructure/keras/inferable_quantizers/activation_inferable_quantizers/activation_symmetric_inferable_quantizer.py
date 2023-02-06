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

from model_compression_toolkit.core.common.constants import FOUND_TF

from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import mark_quantizer, \
    QuantizationTarget


if FOUND_TF:
    import tensorflow as tf
    from model_compression_toolkit.quantizers_infrastructure.keras.inferable_quantizers.activation_inferable_quantizers.activation_uniform_inferable_quantizer import ActivationUniformInferableQuantizer

    @mark_quantizer(quantization_target=QuantizationTarget.Activation,
                    quantization_method=[QuantizationMethod.SYMMETRIC],
                    quantizer_type=None)
    class ActivationSymmetricInferableQuantizer(ActivationUniformInferableQuantizer):

        """
        Class for quantizing activations using a symmetric quantizer
        """

        def __init__(self,
                     num_bits: int,
                     threshold: List[float],
                     signed: bool):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                threshold: threshold for quantizing activations
                signed: whether or not to use signed quantization
            """
            assert isinstance(threshold, list), f'Expected threshold to be of type list but is {type(threshold)}'
            assert all([isinstance(x, (float, np.float32, tf.float32)) for x in
                        threshold]), f'Expected threshold list to contain float or np.float values but found ' \
                                     f'{[type(x) for x in threshold]}'

            self.threshold = np.asarray(threshold)
            self.signed = signed

            delta = self.threshold / (2 ** (num_bits - int(self.signed)))
            # In activation quantization is per-tensor only - thus we pass the threshold as a list with a len of 1
            min_range = list(-self.threshold) if self.signed else [0.0]
            max_range = list(self.threshold - delta)

            super(ActivationSymmetricInferableQuantizer, self).__init__(num_bits=num_bits,
                                                                        min_range=min_range,
                                                                        max_range=max_range)

        def get_config(self):
            """
            Return a dictionary with the configuration of the quantizer.

            Returns:
                Dictionary with the following keys: 'num_bits', 'signed', 'threshold'
            """
            return {'num_bits': self.num_bits,
                    'signed': self.signed,
                    'threshold': self.threshold}


else:
    class ActivationSymmetricInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing tensorflow and tensorflow_model_optimization is mandatory '
                            'when using ActivationSymmetricInferableQuantizer. '
                            'Could not find Tensorflow package.')
