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
"""
Created on 8/13/24

@author: irenab
"""
import torch
from mct_quantizers import mark_quantizer, QuantizationTarget, QuantizationMethod
from mct_quantizers.pytorch.quantizers import ActivationSymmetricInferableQuantizer, ActivationUniformInferableQuantizer

from model_compression_toolkit.gptq.common.gptq_config import ActivationGradProp


class BaseActivationPytorchGPTQTrainableQuantizer:
    """
    Trainable activation quantization for GPTQ.
    It's trainable in a sense that it can be used during training (unlike inferable which do not propagate gradients).
    It does not extend the base trainable quantizer since it doesn't conform to the trainable interface (e.g.
    doesn't have quantization config, which is required in trainable quantizer)
    """
    pass


@mark_quantizer(quantization_target=QuantizationTarget.Activation,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC],
                identifier=ActivationGradProp.STE)
class STEActivationSymmetricGPTQTrainableQuantizer(BaseActivationPytorchGPTQTrainableQuantizer,
                                                   ActivationSymmetricInferableQuantizer):
    """ Symmetric activation quantizer with STE gradient propagation """

    def __call__(self, inputs):
        return torch.fake_quantize_per_tensor_affine(inputs,
                                                     scale=self.scales,
                                                     zero_point=self.zero_points,
                                                     quant_min=self.min_quantized_domain,
                                                     quant_max=self.max_quantized_domain)


@mark_quantizer(quantization_target=QuantizationTarget.Activation,
                quantization_method=[QuantizationMethod.UNIFORM],
                identifier=ActivationGradProp.STE)
class STEActivationUniformGPTQTrainableQuantizer(BaseActivationPytorchGPTQTrainableQuantizer,
                                                 ActivationUniformInferableQuantizer):
    """ Uniform activation quantizer with STE gradient propagation """

    def __call__(self, inputs):
        return torch.fake_quantize_per_tensor_affine(inputs,
                                                     scale=self.scale,
                                                     zero_point=self.zero_point,
                                                     quant_min=self.min_quantized_domain,
                                                     quant_max=self.max_quantized_domain)
