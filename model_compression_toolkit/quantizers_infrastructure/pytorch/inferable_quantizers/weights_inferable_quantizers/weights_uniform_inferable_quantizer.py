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
    import torch
    from model_compression_toolkit.quantizers_infrastructure.pytorch.inferable_quantizers \
        .base_uniform_inferable_quantizer import \
        BaseUniformInferableQuantizer
    from model_compression_toolkit.quantizers_infrastructure.pytorch.quantizer_utils import fix_range_to_include_zero


    class WeightsUniformInferableQuantizer(BaseUniformInferableQuantizer):
        """
        Class for quantizing weights using a uniform quantizer
        """

        def __init__(self,
                     num_bits: int,
                     min_range: torch.Tensor,
                     max_range: torch.Tensor,
                     per_channel: bool,
                     ):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                min_range: min quantization range for quantizing weights
                max_range: max quantization range for quantizing weights
                per_channel: whether to use per-channel quantization
            """
            self.per_channel = per_channel
            min_range, max_range = fix_range_to_include_zero(min_range,
                                                             max_range,
                                                             num_bits)

            # Compute the step size of quantized values.
            self.delta_tensor = (max_range - min_range) / (2 ** num_bits - 1)

            self.max_value = np.reshape(self.max_range, [-1]) if self.per_channel else float(self.max_range)
            self.min_value = np.reshape(self.min_range, [-1]) if self.per_channel else float(self.min_range)

            super(WeightsUniformInferableQuantizer, self).__init__(num_bits,
                                                                   min_range,
                                                                   max_range,
                                                                   QuantizationTarget.Weights)

        def __call__(self,
                     inputs: torch.Tensor,
                     training: bool = True) -> torch.Tensor:
            """
            Weight fake quantizer
            Args:
                inputs: weights to quantize.
                training: whether in training mode or not
            Returns:
                quantized weights
            """

            # Apply rounding
            input_tensor_int = torch.round((inputs - self.min_value) / inputs)

            # Clip data in range
            clipped_tensor = torch.clip(input_tensor_int,
                                        min=0,
                                        max_val=2 ** self.num_bits - 1)

            # Quantize the data between min/max of quantization range.
            q = self.delta_tensor * clipped_tensor + self.min_value
            return q



else:
    class WeightsUniformInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using WeightsUniformInferableQuantizer. '
                            'Could not find torch package.')
