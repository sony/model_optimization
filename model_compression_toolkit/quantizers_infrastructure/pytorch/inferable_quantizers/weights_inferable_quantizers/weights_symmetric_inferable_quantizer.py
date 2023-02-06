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
    import torch
    from model_compression_toolkit.quantizers_infrastructure.pytorch.quantizer_utils import to_torch_tensor, \
        get_working_device
    from model_compression_toolkit.quantizers_infrastructure.pytorch.inferable_quantizers \
        .base_symmetric_inferable_quantizer import \
        BaseSymmetricInferableQuantizer


    @mark_quantizer(quantization_target=QuantizationTarget.Weights,
                    quantization_method=[QuantizationMethod.SYMMETRIC],
                    quantizer_type=None)
    class WeightsSymmetricInferableQuantizer(BaseSymmetricInferableQuantizer):
        """
        Class for quantizing weights using a symmetric quantizer
        """

        def __init__(self,
                     num_bits: int,
                     threshold: np.ndarray,
                     per_channel: bool,
                     channel_axis: int = None
                     ):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                threshold: threshold for quantizing weights
                per_channel: whether to use per-channel quantization
                channel_axis: Axis of input to apply per-channel quantization on.
            """

            super(WeightsSymmetricInferableQuantizer, self).__init__(threshold=threshold,
                                                                     num_bits=num_bits,
                                                                     signed=True)

            if per_channel:
                assert channel_axis is not None, f'Channel axis is missing in per channel quantization'
                assert len(
                    threshold) >= 1, f'In per-channel quantization threshold should be of length >= 1 but is ' \
                                     f'{len(threshold)}'
            else:
                assert len(
                    threshold) == 1, f'In per-tensor quantization threshold should be of length 1 but is {len(threshold)}'

            self.per_channel = per_channel
            self.channel_axis = channel_axis

            self.scales = to_torch_tensor(self.scales).to(get_working_device())
            self.zero_points = torch.zeros(len(threshold), dtype=torch.int32).to(get_working_device())

        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize

            Returns:
                quantized tensor.
            """
            inputs.requires_grad = False
            if self.per_channel:
                return torch.fake_quantize_per_channel_affine(inputs,
                                                              self.scales,
                                                              self.zero_points,
                                                              axis=self.channel_axis,
                                                              quant_min=self.min_quantized_domain,
                                                              quant_max=self.max_quantized_domain)
            return torch.fake_quantize_per_tensor_affine(inputs,
                                                         self.scales,
                                                         self.zero_points,
                                                         quant_min=self.min_quantized_domain,
                                                         quant_max=self.max_quantized_domain)


else:
    class WeightsSymmetricInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using WeightsSymmetricInferableQuantizer. '
                            'Could not find torch package.')
