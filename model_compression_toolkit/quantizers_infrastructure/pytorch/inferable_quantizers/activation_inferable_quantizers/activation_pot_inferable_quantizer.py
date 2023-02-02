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
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import mark_quantizer, \
    QuantizationTarget

if FOUND_TORCH:
    from model_compression_toolkit.quantizers_infrastructure.pytorch.inferable_quantizers\
        .activation_inferable_quantizers.activation_symmetric_inferable_quantizer import \
        ActivationSymmetricInferableQuantizer


    @mark_quantizer(quantization_target=QuantizationTarget.Activation,
                    quantization_method=[QuantizationMethod.POWER_OF_TWO],
                    quantizer_type=None)
    class ActivationPOTInferableQuantizer(ActivationSymmetricInferableQuantizer):
        """
        Class for quantizing activations using power-of-two quantizer
        """

        def __init__(self,
                     num_bits: int,
                     threshold: np.ndarray,
                     signed: bool):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                threshold: threshold for quantizing activations
                signed: whether to use signed quantization or not
            """
            # target of Activation quantization
            super(ActivationPOTInferableQuantizer, self).__init__(num_bits=num_bits,
                                                                  signed=signed,
                                                                  threshold=threshold)

            is_threshold_pot = np.all(np.round(np.log2(threshold.flatten()))==np.log2(threshold.flatten()))
            assert is_threshold_pot, f'Expected threshold to be power of 2 but is {threshold}'


else:
    class ActivationPOTInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using ActivationPOTInferableQuantizer. '
                            'Could not find torch package.')
