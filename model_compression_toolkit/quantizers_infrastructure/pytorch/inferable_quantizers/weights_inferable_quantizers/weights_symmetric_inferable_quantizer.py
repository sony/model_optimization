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
    from model_compression_toolkit.quantizers_infrastructure.pytorch.quantizer_utils import to_torch_tensor
    from model_compression_toolkit.quantizers_infrastructure.pytorch.inferable_quantizers \
        .base_symmetric_inferable_quantizer import \
        BaseSymmetricInferableQuantizer


    class WeightsSymmetricInferableQuantizer(BaseSymmetricInferableQuantizer):
        """
        Class for quantizing weights using a symmetric quantizer
        """

        def __init__(self,
                     num_bits: int,
                     threshold: torch.Tensor,
                     signed: bool,
                     per_channel: bool,
                     ):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                threshold: threshold for quantizing weights
                signed: whether or not to use signed quantization
                per_channel: whether to use per-channel quantization
            """

            super(WeightsSymmetricInferableQuantizer, self).__init__(num_bits=num_bits,
                                                                     threshold=threshold,
                                                                     signed=signed,
                                                                     quantization_target=QuantizationTarget.Weights)

            self.per_channel = per_channel

            delta = self.threshold / np.power(2.0, num_bits - int(signed))
            self.delta_tensor = to_torch_tensor(delta)

            self.min_int = -int(self.signed) * (2 ** (num_bits - int(self.signed)))
            self.max_int = (2 ** (num_bits - int(self.signed))) - 1

        def __call__(self, inputs, training=False):
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize
                training: whether or not the quantizer is being used in training mode (unused here)

            Returns:
                quantized tensor.
            """
            inputs.requires_grad = False
            w0 = torch.round(torch.div(inputs, self.delta_tensor))
            w1 = torch.clip(w0, min=self.min_int, max=self.max_int)
            w_q = self.delta_tensor * w1
            return w_q



else:
    class WeightsSymmetricInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using WeightsSymmetricInferableQuantizer. '
                            'Could not find torch package.')
