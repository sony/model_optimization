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
from model_compression_toolkit.quantizers_infrastructure import QuantizationTarget

if FOUND_TORCH:
    import torch
    from model_compression_toolkit.quantizers_infrastructure.pytorch.quantizer_utils import to_torch_tensor, \
    get_working_device
    from model_compression_toolkit.quantizers_infrastructure.pytorch.inferable_quantizers import BasePyTorchInferableQuantizer



    class WeightsSymmetricInferableQuantizer(BasePyTorchInferableQuantizer):
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
                signed: whether or not to use signed quantization
                per_channel: whether to use per-channel quantization
            """

            super(WeightsSymmetricInferableQuantizer, self).__init__(quantization_target=QuantizationTarget.Weights)

            assert isinstance(threshold,
                              np.ndarray), f'Threshold is expected to be numpy array, but is of type {type(threshold)}'
            assert threshold.ndim == 1, f'Threshold is expected to be flatten, but of shape {threshold.shape}'

            if per_channel:
                assert channel_axis is not None, f'Channel axis is missing in per channel ' \
                                                 f'quantization '
                assert len(
                    threshold) >= 1, f'In per-channel quantization threshold should ' \
                                     f'be of length >= 1 but is {len(threshold)}'
            else:
                assert len(threshold) == 1, f'In per-tensor quantization threshold should ' \
                                     f'be of length 1 but is {len(threshold)}'

            # TODO: assert that channel axis is in valid range
            self.num_bits = num_bits
            self.threshold = threshold
            self.per_channel = per_channel
            self.channel_axis = channel_axis


            scales = self.threshold / np.power(2.0, num_bits - 1)

            self.scales = to_torch_tensor(scales).to(get_working_device())
            self.zero_points = torch.zeros(len(threshold), dtype=torch.int32).to(get_working_device())

            # Integers. Min and max quantiation domain - here, we always use signed quantization
            self.min_quantized_domain = -2 ** (self.num_bits - 1)
            self.max_quantized_domain = 2 ** (self.num_bits - 1) - 1

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
                # return torch.quantize_per_channel(inputs,
                #                                   self.scales,
                #                                   self.zero_points,
                #                                   axis=self.channel_axis,
                #                                   dtype=self.quantization_dtype)
            # return torch.quantize_per_tensor(inputs,
            #                                  self.scales,
            #                                  self.zero_points,
            #                                  self.quantization_dtype)


            # w0 = torch.round(torch.div(inputs, self.delta_tensor))
            # w1 = torch.clip(w0, min=self.min_int, max=self.max_int)
            # w_q = self.delta_tensor * w1
            # return w_q



else:
    class WeightsSymmetricInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using WeightsSymmetricInferableQuantizer. '
                            'Could not find torch package.')
