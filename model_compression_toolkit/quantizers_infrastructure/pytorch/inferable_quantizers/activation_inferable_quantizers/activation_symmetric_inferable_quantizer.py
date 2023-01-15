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
    from model_compression_toolkit.quantizers_infrastructure.pytorch.quantizer_utils import \
        get_activation_symmetric_quantization_range_and_scale
    from model_compression_toolkit.quantizers_infrastructure.pytorch.inferable_quantizers \
        .base_symmetric_inferable_quantizer import \
        BaseSymmetricInferableQuantizer


    class ActivationSymmetricInferableQuantizer(BaseSymmetricInferableQuantizer):
        """
        Class for quantizing activations using a symmetric quantizer
        """

        def __init__(self,
                     num_bits: int,
                     threshold: float,
                     signed: bool):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                threshold: threshold for quantizing activations
                signed: whether or not to use signed quantization
            """

            super(ActivationSymmetricInferableQuantizer, self).__init__(num_bits=num_bits,
                                                                        threshold=threshold,
                                                                        signed=signed,
                                                                        quantization_target=QuantizationTarget.Activation)

            self.min_range, self.max_range, self.scale = get_activation_symmetric_quantization_range_and_scale(
                activation_is_signed=signed,
                activation_n_bits=num_bits,
                activation_threshold=threshold)

            self.zero_point = 0

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
    class ActivationSymmetricInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using ActivationSymmetricInferableQuantizer. '
                            'Could not find torch package.')
