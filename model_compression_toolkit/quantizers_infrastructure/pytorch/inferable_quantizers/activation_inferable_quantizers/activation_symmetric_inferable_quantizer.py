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
    from model_compression_toolkit.quantizers_infrastructure.pytorch.quantizer_utils import \
    get_activation_symmetric_quantization_range_and_scale, to_torch_tensor
    from model_compression_toolkit.quantizers_infrastructure.pytorch.inferable_quantizers import BasePyTorchInferableQuantizer


    class ActivationSymmetricInferableQuantizer(BasePyTorchInferableQuantizer):
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
                signed: whether or not to use signed quantization
            """

            super(ActivationSymmetricInferableQuantizer, self).__init__(quantization_target=QuantizationTarget.Activation)
            assert isinstance(threshold,
                              np.ndarray), f'Threshold is expected to be numpy array, but is of type {type(threshold)}'
            assert threshold.ndim == 1, f'Threshold is expected to be flatten, but of shape {threshold.shape}'

            assert len(threshold)==1, f'For activation, quantization per channel is not supported and threshold should be of length 1 but is {len(threshold)}'
            threshold = threshold[0]

            self.min_quantized_domain, self.max_quantized_domain, self.scales = get_activation_symmetric_quantization_range_and_scale(
                activation_is_signed=signed,
                activation_n_bits=num_bits,
                activation_threshold=threshold)

            # self.scales = to_torch_tensor(self.scales)
            # assert self.scales.dim()==1
            self.zero_points = 0 #torch.Tensor([0]).int()

        def __call__(self, inputs: torch.Tensor):
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize

            Returns:
                quantized tensor.
            """

            # self.scales = self.scales.to(inputs.device)
            # self.zero_points = self.zero_points.to(inputs.device)
            # print('inputs.device : ', inputs.device)
            # print('scales.device : ', self.scales.device)
            # print('zero_points.device : ', self.zero_points.device)
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
