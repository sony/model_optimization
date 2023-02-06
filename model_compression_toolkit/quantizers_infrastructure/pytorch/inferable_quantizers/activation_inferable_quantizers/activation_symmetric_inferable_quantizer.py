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
        .base_symmetric_inferable_quantizer import \
        BaseSymmetricInferableQuantizer


    @mark_quantizer(quantization_target=QuantizationTarget.Activation,
                    quantization_method=[QuantizationMethod.SYMMETRIC],
                    quantizer_type=None)
    class ActivationSymmetricInferableQuantizer(BaseSymmetricInferableQuantizer):
        """
        Class for quantizing activations using a symmetric quantizer
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

            super(ActivationSymmetricInferableQuantizer, self).__init__(
                num_bits=num_bits,
                threshold=threshold,
                signed=signed)

            # Activation supports only per-tensor quantization
            assert len(
                self.scales) == 1, f'For activation, quantization per channel is not supported and threshold should ' \
                                   f'be of length 1 but is {len(threshold)}'
            self.scales = self.scales[0]

            self.zero_points = 0

        def __call__(self, inputs: torch.Tensor):
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize

            Returns:
                quantized tensor.
            """
            return torch.fake_quantize_per_tensor_affine(inputs,
                                                         scale=self.scales,
                                                         zero_point=self.zero_points,
                                                         quant_min=self.min_quantized_domain,
                                                         quant_max=self.max_quantized_domain)

else:
    class ActivationSymmetricInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using ActivationSymmetricInferableQuantizer. '
                            'Could not find torch package.')
