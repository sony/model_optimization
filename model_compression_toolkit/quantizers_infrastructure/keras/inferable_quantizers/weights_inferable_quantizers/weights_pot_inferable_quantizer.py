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
    from model_compression_toolkit.quantizers_infrastructure.keras.inferable_quantizers.weights_inferable_quantizers.weights_symmetric_inferable_quantizer import WeightsSymmetricInferableQuantizer

    @mark_quantizer(quantization_target=QuantizationTarget.Weights,
                    quantization_method=[QuantizationMethod.POWER_OF_TWO],
                    quantizer_type=None)
    class WeightsPOTInferableQuantizer(WeightsSymmetricInferableQuantizer):
        """
        Class for quantizing weights using power-of-two quantizer
        """

        def __init__(self,
                     num_bits: int,
                     threshold: List[float],
                     per_channel: bool,
                     channel_axis: int = None,
                     input_rank: int = None):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                threshold: threshold for quantizing activations
                per_channel: whether to use per-channel quantization
                channel_axis: axis along which to apply per-channel quantization
                input_rank: number of dimensions of input tensor the quantizer quantizes
            """
            # Call the superclass constructor with the given parameters, along with the target of Weights quantization
            super(WeightsPOTInferableQuantizer, self).__init__(num_bits=num_bits,
                                                               threshold=threshold,
                                                               per_channel=per_channel,
                                                               channel_axis=channel_axis,
                                                               input_rank=input_rank)

            is_threshold_pot = np.all([int(np.log2(x)) == np.log2(x) for x in self.threshold.flatten()])
            assert is_threshold_pot, f'Expected threshold to be power of 2 but is {self.threshold}'


else:
    class WeightsPOTInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing tensorflow and tensorflow_model_optimization is mandatory '
                            'when using WeightsPOTInferableQuantizer. '
                            'Could not find Tensorflow package.')
