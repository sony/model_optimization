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
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import mark_quantizer

if FOUND_TORCH:
    from model_compression_toolkit.quantizers_infrastructure.pytorch.inferable_quantizers\
        .base_pytorch_inferable_quantizer import \
        BasePyTorchInferableQuantizer


    @mark_quantizer(quantization_target=None,
                    quantization_method=[QuantizationMethod.SYMMETRIC],
                    quantizer_type=None)
    class BaseSymmetricInferableQuantizer(BasePyTorchInferableQuantizer):

        def __init__(self,
                     num_bits: int,
                     threshold: np.ndarray,
                     signed: bool):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                threshold: threshold for quantizing weights
                signed: whether or not to use signed quantization
            """

            super(BaseSymmetricInferableQuantizer, self).__init__()

            assert isinstance(threshold,
                              np.ndarray), f'Threshold is expected to be numpy array, but is of type {type(threshold)}'
            assert threshold.ndim == 1, f'Threshold is expected to be flatten, but of shape {threshold.shape}'

            self.signed = signed
            self.threshold = threshold
            self.num_bits = num_bits

            if signed:
                self.min_quantized_domain = -2 ** (num_bits - 1)
                self.max_quantized_domain = 2 ** (num_bits - 1) - 1
                self.scales = threshold / 2 ** (num_bits - 1)
            else:
                self.min_quantized_domain = 0
                self.max_quantized_domain = (2 ** num_bits) - 1
                self.scales = threshold / 2 ** num_bits



else:
    class BaseSymmetricInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using BaseSymmetricInferableQuantizer. '
                            'Could not find torch package.')
