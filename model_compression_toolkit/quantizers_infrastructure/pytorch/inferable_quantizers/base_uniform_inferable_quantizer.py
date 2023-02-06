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
                    quantization_method=[QuantizationMethod.UNIFORM],
                    quantizer_type=None)
    class BaseUniformInferableQuantizer(BasePyTorchInferableQuantizer):

        def __init__(self,
                     num_bits: int,
                     min_range: np.ndarray,
                     max_range: np.ndarray):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                min_range: min quantization range for quantizing
                max_range: max quantization range for quantizing
            """

            super(BaseUniformInferableQuantizer, self).__init__()
            self.num_bits = num_bits
            self.min_range = min_range
            self.max_range = max_range
            self.min_quantized_domain = 0
            self.max_quantized_domain = 2 ** num_bits - 1


else:
    class BaseUniformInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using BaseUniformInferableQuantizer. '
                            'Could not find torch package.')
