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
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from model_compression_toolkit.qat import TrainingMethod
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from mct_quantizers import PytorchQuantizationWrapper
from model_compression_toolkit.qat.common import THRESHOLD_TENSOR
from model_compression_toolkit import constants as C
from model_compression_toolkit.qat.pytorch.quantizer.base_pytorch_qat_quantizer import BasePytorchQATTrainableQuantizer
from mct_quantizers.common.base_inferable_quantizer import mark_quantizer, QuantizationTarget

from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.qat.pytorch.quantizer.quantizer_utils import ste_round, grad_scale
from mct_quantizers.pytorch.quantizers import \
    WeightsPOTInferableQuantizer, WeightsSymmetricInferableQuantizer, ActivationPOTInferableQuantizer, \
    ActivationSymmetricInferableQuantizer
from model_compression_toolkit.trainable_infrastructure.common.trainable_quantizer_config import \
    TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup


def symmetric_lsq_quantizer(x: nn.Parameter,
                          thresholds: nn.Parameter,
                          num_bits: int,
                          sign: bool,
                          min_int: int,
                          max_int: int,
                          scale_factor: float) -> Union[nn.Parameter, torch.Tensor]:
    """
    Symmetric quantizer according to LSQ algorithm: https://arxiv.org/pdf/1902.08153.pdf
    Args:
        x: input to quantize
        thresholds: thresholds of quantization levels
        num_bits: number of bits for quantization
        sign: whether x is signed or not
        min_int: min clipping integer value
        max_int: max clipping integer value
        scale_factor: grad scale of LSQ algorithm
    Returns:
        A quantized tensor
    """
    delta = thresholds / (2 ** (num_bits - int(sign)))
    delta_scaled = grad_scale(delta, scale_factor)
    rounded = ste_round(x / delta_scaled)
    clipped = torch.clip(rounded, min=min_int, max=max_int)
    quantized = delta_scaled * clipped
    return quantized


@mark_quantizer(quantization_target=QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC],
                identifier=TrainingMethod.LSQ)
class LSQWeightQATQuantizer(BasePytorchQATTrainableQuantizer):
    """
    Trainable constrained quantizer to quantize layer's weights.
    """

    def __init__(self, quantization_config: TrainableQuantizerWeightsConfig):
        """
        Initialize a LSQWeightQATQuantizer object with parameters to use
        for the quantization.

        Args:
            quantization_config: trainable quantizer config class
        """
        super().__init__(quantization_config)
        self.power_of_two = quantization_config.weights_quantization_method == QuantizationMethod.POWER_OF_TWO
        self.threshold_values = np.array(quantization_config.weights_quantization_params[C.THRESHOLD])
        if self.power_of_two:
            self.threshold_values = np.power(2.0, np.ceil(np.log2(np.maximum(self.threshold_values, C.MIN_THRESHOLD))))
        self.num_bits = self.quantization_config.weights_n_bits
        n_pos_bits = self.num_bits - int(C.WEIGHTS_SIGNED)
        self.min_int = -int(C.WEIGHTS_SIGNED) * (2 ** n_pos_bits)
        self.max_int = 2 ** n_pos_bits - 1
        self.scale_factor = 1.0 / np.sqrt(self.max_int * self.threshold_values.size)

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

        # Add threshold variables to layer.
        layer.register_parameter(name + "_" + THRESHOLD_TENSOR, nn.Parameter(to_torch_tensor(self.threshold_values), requires_grad=True))

        # save the quantizer added parameters for later calculations
        self.add_quantizer_variable(THRESHOLD_TENSOR, layer.get_parameter(name + "_" + THRESHOLD_TENSOR), VariableGroup.QPARAMS)


    def __call__(self,
                 inputs: nn.Parameter,
                 training: bool) -> nn.Parameter:
        """
        Quantize a tensor
        Args:
            inputs: Input tensor to quantize.
            training: whether in training mode or not
        Returns:
            quantized tensor
        """
        thresholds = self.get_quantizer_variable(THRESHOLD_TENSOR)
        weight_quantized = symmetric_lsq_quantizer(inputs, thresholds, self.num_bits, C.WEIGHTS_SIGNED, self.min_int, self.max_int, self.scale_factor)
        return weight_quantized

    def convert2inferable(self) -> Union[WeightsPOTInferableQuantizer, WeightsSymmetricInferableQuantizer]:
        """
        Convert quantizer to inferable quantizer.

        Returns:
            A pytorch inferable quanizer object.
        """
        threshold_values = self.get_quantizer_variable(THRESHOLD_TENSOR).cpu().detach().numpy().flatten()
        if self.power_of_two:
            pot_threshold = 2 ** np.ceil(np.log2(threshold_values))
            return WeightsPOTInferableQuantizer(num_bits=self.num_bits,
                                                threshold=pot_threshold.tolist(),
                                                per_channel=self.quantization_config.weights_per_channel_threshold,
                                                channel_axis=self.quantization_config.weights_channels_axis)
        else:
            return WeightsSymmetricInferableQuantizer(num_bits=self.num_bits,
                                                      threshold=threshold_values.tolist(),
                                                      per_channel=self.quantization_config.weights_per_channel_threshold,
                                                      channel_axis=self.quantization_config.weights_channels_axis)



@mark_quantizer(quantization_target=QuantizationTarget.Activation,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC],
                identifier=TrainingMethod.LSQ)
class LSQActivationQATQuantizer(BasePytorchQATTrainableQuantizer):
    """
    Trainable constrained quantizer to quantize layer activations.
    """

    def __init__(self, quantization_config: TrainableQuantizerActivationConfig):
        """
        Initialize a LSQActivationQATQuantizer object with parameters to use
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
