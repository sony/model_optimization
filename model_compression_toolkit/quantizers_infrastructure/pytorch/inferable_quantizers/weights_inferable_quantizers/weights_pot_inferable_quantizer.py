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

from model_compression_toolkit.core.common.constants import FOUND_TORCH
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import QuantizationTarget

if FOUND_TORCH:
    from model_compression_toolkit.quantizers_infrastructure.pytorch.inferable_quantizers.weights_inferable_quantizers.weights_symmetric_inferable_quantizer import \
        WeightsSymmetricInferableQuantizer

    class WeightsPOTInferableQuantizer(WeightsSymmetricInferableQuantizer):
        """
        Class for quantizing weights using power-of-two quantizer
        """

        def __init__(self,
                     num_bits: int,
                     threshold: np.ndarray,
                     signed: bool,
                     per_channel: bool,
                     ):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                threshold: threshold for quantizing activations
                signed: whether or not to use signed quantization
                per_channel: whether to use per-channel quantization
            """

            is_threshold_pot = np.all([int(np.log2(x)) == np.log2(x) for x in threshold.flatten()])
            assert is_threshold_pot, f'Expected threshold to be power of 2 but is {threshold}'

            # target of Weights quantization
            super(WeightsPOTInferableQuantizer, self).__init__(num_bits=num_bits,
                                                               threshold=threshold,
                                                               signed=signed,
                                                               per_channel=per_channel)





else:
    class WeightsPOTInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using WeightsPOTInferableQuantizer. '
                            'Could not find torch package.')
