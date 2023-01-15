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
        .base_uniform_inferable_quantizer import \
        BaseUniformInferableQuantizer


    class ActivationUniformInferableQuantizer(BaseUniformInferableQuantizer):
        """
        Class for quantizing activations using an uniform quantizer
        """

        def __init__(self,
                     num_bits: int,
                     min_range: torch.Tensor,
                     max_range: torch.Tensor,
                     ):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                min_range: min range for quantizing activations
                max_range: max range for quantizing activations
            """
            super(ActivationUniformInferableQuantizer, self).__init__(num_bits,
                                                                      min_range,
                                                                      max_range,
                                                                      QuantizationTarget.Activation)

            # fixing quantization range to include 0
            a = 0 if min_range > 0 else min_range
            b = 0 if max_range < 0 else max_range

            self.min_range = 0
            self.max_range = 2 ** num_bits - 1

            self.scale = (b - a) / ((2 ** num_bits) - 1)
            self.zero_point = -int(a / self.scale)  # zp has to be positive, and a <=0, so we multiply by -1

        def __call__(self, inputs, training=False):
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize
                training: whether or not the quantizer is being used in training mode (unused here)

            Returns:
                quantized tensor.
            """
            return torch.fake_quantize_per_tensor_affine(inputs,
                                                         scale=self.scale,
                                                         zero_point=self.zero_point,
                                                         quant_min=self.min_range,
                                                         quant_max=self.max_range)


else:
    class ActivationUniformInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using ActivationUniformInferableQuantizer. '
                            'Could not find torch package.')
