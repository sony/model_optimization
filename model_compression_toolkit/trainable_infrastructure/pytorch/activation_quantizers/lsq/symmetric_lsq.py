# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Union

import numpy as np
import torch
from torch import nn

from mct_quantizers import mark_quantizer, QuantizationTarget, QuantizationMethod, PytorchQuantizationWrapper
from mct_quantizers.pytorch.quantizers import ActivationPOTInferableQuantizer, ActivationSymmetricInferableQuantizer
from model_compression_toolkit import constants as C
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerActivationConfig, TrainingMethod
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup
from model_compression_toolkit.trainable_infrastructure.common.constants import THRESHOLD_TENSOR
from model_compression_toolkit.trainable_infrastructure.pytorch.activation_quantizers import \
    BasePytorchActivationTrainableQuantizer
from model_compression_toolkit.trainable_infrastructure.pytorch.quantizer_utils import symmetric_lsq_quantizer


# moved (and renamed) from model_compression_toolkit/qat/pytorch/quantizer/lsq/symmetric_lsq.py
@mark_quantizer(quantization_target=QuantizationTarget.Activation,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC],
                identifier=TrainingMethod.LSQ)
class LSQSymmetricActivationTrainableQuantizer(BasePytorchActivationTrainableQuantizer):
    """
    Trainable constrained quantizer to quantize layer activations.
    """

    def __init__(self, quantization_config: TrainableQuantizerActivationConfig):
        """
        Initialize a LSQSymmetricActivationTrainableQuantizer object with parameters to use
        for symmetric or power of two quantization.

        Args:
            quantization_config: trainable quantizer config class
        """
        super().__init__(quantization_config)
        self.power_of_two = quantization_config.activation_quantization_method == QuantizationMethod.POWER_OF_TWO
        self.sign = quantization_config.activation_quantization_params['is_signed']
        self.threshold_values = np.array([quantization_config.activation_quantization_params[C.THRESHOLD]])
        self.num_bits = quantization_config.activation_n_bits
        n_pos_bits = self.num_bits - int(self.sign)
        self.min_int = -int(self.sign) * (2 ** n_pos_bits)
        self.max_int = (2 ** n_pos_bits) - 1

    def initialize_quantization(self,
                                tensor_shape: torch.Size,
                                name: str,
                                layer: PytorchQuantizationWrapper):
        """
        Add quantizer parameters to the quantizer parameters dictionary

        Args:
            tensor_shape: tensor shape of the quantized tensor.
            name: Tensor name.
            layer: Layer to quantize.
        """
        layer.register_parameter(name, nn.Parameter(to_torch_tensor(self.threshold_values), requires_grad=True))

        # save the quantizer added parameters for later calculations
        self.add_quantizer_variable(THRESHOLD_TENSOR, layer.get_parameter(name), VariableGroup.QPARAMS)

    def __call__(self,
                 inputs: torch.Tensor,
                 training: bool = True) -> torch.Tensor:
        """
        Quantize a tensor.
        Args:
            inputs: Input tensor to quantize.
            training: Whether the graph is in training mode.

        Returns:
            The quantized tensor.
        """

        thresholds = self.get_quantizer_variable(THRESHOLD_TENSOR)
        n_channels = inputs.shape[1]
        scale_factor = 1.0 / np.sqrt(self.max_int * n_channels)
        inputs_quantized = symmetric_lsq_quantizer(inputs, thresholds, self.num_bits, self.sign, self.min_int, self.max_int, scale_factor)
        return inputs_quantized

    def convert2inferable(self) -> Union[ActivationPOTInferableQuantizer, ActivationSymmetricInferableQuantizer]:
        """
        Convert quantizer to inferable quantizer.

        Returns:
            A pytorch inferable quanizer object.
        """
        threshold_values = self.get_quantizer_variable(THRESHOLD_TENSOR).cpu().detach().numpy()
        if self.power_of_two:
            pot_threshold = np.power(2.0, np.ceil(np.log2(threshold_values)))
            return ActivationPOTInferableQuantizer(num_bits=self.num_bits,
                                                   threshold=pot_threshold.tolist(),
                                                   signed=self.sign)
        else:
            return ActivationSymmetricInferableQuantizer(num_bits=self.num_bits,
                                                         threshold=threshold_values.tolist(),
                                                         signed=self.sign)
