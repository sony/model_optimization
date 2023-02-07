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
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import mark_quantizer, QuantizationTarget

if FOUND_TF:
    from model_compression_toolkit.quantizers_infrastructure.keras.inferable_quantizers.weights_inferable_quantizers.weights_uniform_inferable_quantizer import WeightsUniformInferableQuantizer

    @mark_quantizer(quantization_target=QuantizationTarget.Weights,
                    quantization_method=[QuantizationMethod.SYMMETRIC],
                    quantizer_type=None)
    class WeightsSymmetricInferableQuantizer(WeightsUniformInferableQuantizer):
        """
        Class for quantizing weights using a symmetric quantizer
        """
        def __init__(self,
                     num_bits: int,
                     threshold: List[float],
                     per_channel: bool,
                     channel_axis: int = None,
                     input_rank: int = None
                     ):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                threshold: threshold for quantizing weights
                per_channel: whether to use per-channel quantization
                channel_axis: axis along which to apply per-channel quantization
                input_rank: number of dimensions of input tensor the quantizer quantizes
            """
            assert isinstance(threshold, list), f'Expected threshold to be of type list but is {type(threshold)}'
            assert all([isinstance(x, (float, np.float32, np.float64)) for x in
                        threshold]), f'Expected threshold list to contain float or np.float values but found ' \
                                     f'{[type(x) for x in threshold]}'

            self.threshold = np.asarray(threshold)

            _min_range = -self.threshold
            _max_range = self.threshold - self.threshold / (2 ** (num_bits - 1))

            super(WeightsSymmetricInferableQuantizer, self).__init__(num_bits=num_bits,
                                                                     min_range=list(_min_range),
                                                                     max_range=list(_max_range),
                                                                     per_channel=per_channel,
                                                                     channel_axis=channel_axis,
                                                                     input_rank=input_rank)

        def get_config(self):
            """
            Return a dictionary with the configuration of the quantizer.

            Returns:
                Dictionary with the following keys: 'num_bits', 'threshold', 'per_channel', 'channel_axis'
            """
            return {'num_bits': self.num_bits,
                    'threshold': self.threshold,
                    'per_channel': self.per_channel,
                    'channel_axis': self.channel_axis,
                    'input_rank': self.input_rank}

else:
    class WeightsSymmetricInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing tensorflow and tensorflow_model_optimization is mandatory '
                            'when using WeightsPOTInferableQuantizer. '
                            'Could not find Tensorflow package.')
