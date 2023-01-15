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

from model_compression_toolkit.core.common.constants import FOUND_TORCH
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import QuantizationTarget


if FOUND_TORCH:
    import torch
    from model_compression_toolkit.quantizers_infrastructure.pytorch.inferable_quantizers\
        .base_uniform_inferable_quantizer \
        import \
        BaseUniformInferableQuantizer

    class BaseSymmetricInferableQuantizer(BaseUniformInferableQuantizer):

        def __init__(self,
                     num_bits: int,
                     threshold: torch.Tensor,
                     signed: bool,
                     quantization_target: QuantizationTarget):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                threshold: threshold for quantizing weights
                signed: whether or not to use signed quantization
                quantization_target: An enum which selects the quantizer tensor type: activation or weights.
            """

            self.signed = signed
            self.threshold = threshold

            delta = threshold / (2 ** (num_bits - int(signed)))
            min_range = -threshold if signed else 0
            max_range = threshold - delta

            super(BaseSymmetricInferableQuantizer, self).__init__(quantization_target=quantization_target,
                                                                  min_range=min_range,
                                                                  max_range=max_range,
                                                                  num_bits=num_bits)

else:
    class BaseSymmetricInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using BaseSymmetricInferableQuantizer. '
                            'Could not find torch package.')
