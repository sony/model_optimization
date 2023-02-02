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
    from model_compression_toolkit.quantizers_infrastructure.pytorch.inferable_quantizers \
        .base_uniform_inferable_quantizer import \
        BaseUniformInferableQuantizer


    @mark_quantizer(quantization_target=QuantizationTarget.Activation,
                    quantization_method=[QuantizationMethod.UNIFORM],
                    quantizer_type=None)
    class ActivationUniformInferableQuantizer(BaseUniformInferableQuantizer):
        """
        Class for quantizing activations using an uniform quantizer
        """

        def __init__(self,
                     num_bits: int,
                     min_range: np.ndarray,
                     max_range: np.ndarray,
                     ):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                min_range: min range for quantizing activations
                max_range: max range for quantizing activations
            """
            super(ActivationUniformInferableQuantizer, self).__init__(num_bits=num_bits,
                                                                      min_range=min_range,
                                                                      max_range=max_range)

            assert isinstance(min_range,
                              np.ndarray), f'min_range is expected to be numpy array, but is of type {type(min_range)}'
            assert isinstance(max_range,
                              np.ndarray), f'max_range is expected to be numpy array, but is of type {type(max_range)}'
            assert min_range.ndim == 1, f'min_range is expected to be flatten, but of shape {min_range.shape}'
            assert max_range.ndim == 1, f'max_range is expected to be flatten, but of shape {min_range.shape}'

            assert len(
                min_range) == 1, f'For activation, quantization per channel is not supported and min_range should be ' \
                                 f'of length 1 but is {len(min_range)}'
            assert len(
                max_range) == 1, f'For activation, quantization per channel is not supported and max_range should be ' \
                                 f'of length 1 but is {len(max_range)}'

            # Activation is per-tensor thus we expect only a single min/max values
            min_range = min_range[0]
            max_range = max_range[0]

            # fixing quantization range to include 0
            a = 0 if min_range > 0 else min_range
            b = 0 if max_range < 0 else max_range

            self.scale = float((b - a) / ((2 ** num_bits) - 1))
            self.zero_point = int(-np.round(a / self.scale))  # zp has to be positive, and a <=0, so we multiply by -1

        def __call__(self, inputs: torch.Tensor):
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize

            Returns:
                quantized tensor.
            """
            return torch.fake_quantize_per_tensor_affine(inputs,
                                                         scale=self.scale,
                                                         zero_point=self.zero_point,
                                                         quant_min=self.min_quantized_domain,
                                                         quant_max=self.max_quantized_domain)


else:
    class ActivationUniformInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using ActivationUniformInferableQuantizer. '
                            'Could not find torch package.')
