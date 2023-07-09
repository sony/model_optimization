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
import torch
import torch.nn as nn
from typing import Dict
import numpy as np

from model_compression_toolkit.core.common import max_power_of_two
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from mct_quantizers import QuantizationTarget, PytorchQuantizationWrapper
from model_compression_toolkit.gptq.common.gptq_config import RoundingType
from model_compression_toolkit.gptq.pytorch.quantizer.base_pytorch_gptq_quantizer import \
    BasePytorchGPTQTrainableQuantizer
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, torch_tensor_to_numpy
from model_compression_toolkit.gptq.pytorch.quantizer import quant_utils as qutils
from model_compression_toolkit.gptq.common.gptq_constants import PTQ_THRESHOLD, SCALE_PTQ, \
    SOFT_ROUNDING_GAMMA, SOFT_ROUNDING_ZETA, AUXVAR
from model_compression_toolkit.constants import THRESHOLD, MIN_THRESHOLD
from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerWeightsConfig
from mct_quantizers import mark_quantizer
from model_compression_toolkit.trainable_infrastructure.common.quant_utils import \
    get_threshold_reshape_shape
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup


def soft_rounding_symmetric_quantizer(input_tensor: torch.Tensor,
                                      auxvar_tensor: torch.Tensor,
                                      threshold_tensor: torch.Tensor,
                                      num_bits: int,
                                      signed: bool,
                                      power_of_two: bool) -> torch.Tensor:
    """
    Quantize a tensor symmetrically for GPTQ quantizers.

    Args:
        input_tensor: Tensor to quantize. values of this tensor are not changed during gptq.
        auxvar_tensor: Tensor that manifests the bit shift of the quantized weights due to gptq training.
        threshold_tensor: Tensor with values to compute the threshold.
        num_bits: Num of bits to use.
        signed: Signedness of the quantization range.
        power_of_two: Whether the threshold should be constrained or not.

    Returns:
        A quantized tensor.
    """

    if power_of_two:
        threshold_tensor = qutils.power_of_two_max(threshold_tensor)
    delta = qutils.calculate_delta(threshold_tensor, num_bits, signed)
    with torch.no_grad():
        input_tensor_int = torch.floor(input_tensor / delta)
    tensor_q = input_tensor_int + auxvar_tensor
    int_threshold = 2 ** (num_bits - int(signed))
    return delta * qutils.ste_clip(tensor_q,
                                   min_val=-int(signed) * int_threshold,
                                   max_val=int_threshold - 1)


@mark_quantizer(quantization_target=QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC],
                identifier=RoundingType.SoftQuantizer)
class SymmetricSoftRoundingGPTQ(BasePytorchGPTQTrainableQuantizer):
    """
    Trainable symmetric quantizer to optimize the rounding of the quantized values using a soft quantization method.
    """

    def __init__(self,
                 quantization_config: TrainableQuantizerWeightsConfig,
                 quantization_parameter_learning: bool = False):
        """
        Construct a Pytorch model that utilize a fake weight quantizer of soft-quantizer for symmetric quantizer.

        Args:
            quantization_config: Trainable weights quantizer config.
            quantization_parameter_learning (Bool): Whether to learn the threshold or not
        """

        super().__init__(quantization_config)
        self.num_bits = quantization_config.weights_n_bits
        self.per_channel = quantization_config.weights_per_channel_threshold

        threshold_values = quantization_config.weights_quantization_params[THRESHOLD]
        self.threshold_shape = np.asarray(threshold_values).shape
        self.threshold_values = np.reshape(np.asarray(threshold_values), [-1]) if self.per_channel else float(
            threshold_values)

        self.quantization_axis = quantization_config.weights_channels_axis
        self.power_of_two = quantization_config.weights_quantization_method == QuantizationMethod.POWER_OF_TWO
        self.quantization_parameter_learning = quantization_parameter_learning

        # gamma and zeta are stretch parameters for computing the rectified sigmoind function.
        # See: https://arxiv.org/pdf/2004.10568.pdf
        self.gamma = SOFT_ROUNDING_GAMMA
        self.zeta = SOFT_ROUNDING_ZETA

        self.quantizer_parameters = {}

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

        if self.per_channel:
            threshold_tensor = to_torch_tensor(self.threshold_values)
        else:
            threshold_tensor = torch.tensor(self.threshold_values)
        layer.register_parameter(f"{name}_{PTQ_THRESHOLD}",
                                 nn.Parameter(threshold_tensor, requires_grad=False))

        w = layer.layer.weight
        delta = qutils.calculate_delta(threshold_tensor.reshape(self.threshold_shape), self.num_bits, signed=True)
        w_clipped_normed = torch.clip(w / delta, -2**(self.num_bits-1), 2**(self.num_bits-1)-1)
        rest = w_clipped_normed - torch.floor(w_clipped_normed)  # rest of rounding [0, 1)
        # Note that (rest - self.gamma) can't be zero since rest is positive and gamma is negative, so the division
        # is safe
        alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest

        layer.register_parameter(f"{name}_{AUXVAR}", nn.Parameter(alpha, requires_grad=True))

        # save the quantizer added parameters for later calculations
        self.add_quantizer_variable(PTQ_THRESHOLD, layer.get_parameter(f"{name}_{PTQ_THRESHOLD}"), VariableGroup.QPARAMS)
        self.add_quantizer_variable(AUXVAR, layer.get_parameter(f"{name}_{AUXVAR}"), VariableGroup.WEIGHTS)

        if self.quantization_parameter_learning:
            if self.per_channel:
                layer.register_parameter(f"{name}_{SCALE_PTQ}",
                                         nn.Parameter(to_torch_tensor(torch.ones_like(torch.Tensor(self.threshold_values))),
                                                      requires_grad=True))
            else:
                layer.register_parameter(f"{name}_{SCALE_PTQ}",
                                         nn.Parameter(to_torch_tensor((torch.tensor([1.0], requires_grad=True)))))
            self.add_quantizer_variable(SCALE_PTQ, layer.get_parameter(f"{name}_{SCALE_PTQ}"), VariableGroup.QPARAMS)

    def get_soft_targets(self) -> torch.Tensor:
        """
        Computes the rectified sigmoid function for the quantization target parameters.

        Returns:
            A tensor with the soft rounding targets values.

        """
        scaled_sigmoid = torch.sigmoid(self.get_quantizer_variable(AUXVAR)) * (self.zeta - self.gamma) + self.gamma
        return torch.clip(scaled_sigmoid, min=0, max=1)

    def get_quant_config(self) -> Dict[str, np.ndarray]:
        """
        Returns the config used to edit NodeQuantizationConfig after GPTQ retraining

        Returns:
            A dictionary of attributes the quantize_config retraining has changed during GPTQ retraining.
            Keys must match NodeQuantizationConfig attributes

        """
        old_threshold = torch_tensor_to_numpy(self.get_quantizer_variable(PTQ_THRESHOLD))
        old_threshold = np.resize(old_threshold, self.threshold_shape)
        if self.power_of_two:
            old_threshold = max_power_of_two(old_threshold, MIN_THRESHOLD)
        else:
            if self.quantization_parameter_learning:
                scale = torch.reshape(self.get_quantizer_variable(SCALE_PTQ), self.threshold_shape)
                scale = torch.where(scale <= 0, torch.tensor(MIN_THRESHOLD, device=scale.device), scale)
                old_threshold = old_threshold * torch_tensor_to_numpy(scale)
        old_threshold = old_threshold.reshape(self.threshold_shape)
        return {THRESHOLD: old_threshold}

    def __call__(self,
                 inputs: nn.Parameter,
                 training: bool) -> torch.Tensor:
        """
        Quantize a tensor.

        Args:
            inputs: Input tensor to quantize.
            training: whether in training mode or not

        Returns:
            quantized tensor
        """
        auxvar = self.get_quantizer_variable(AUXVAR)
        ptq_threshold_tensor = self.get_quantizer_variable(PTQ_THRESHOLD)

        #####################################################
        # Soft Rounding
        #####################################################
        aux_var = self.get_soft_targets()
        if not training:
            aux_var = (aux_var >= 0.5).to(auxvar.dtype)

        if self.per_channel:
            reshape_shape = get_threshold_reshape_shape(inputs.shape,
                                                        quant_axis=self.quantization_axis,
                                                        quant_axis_dim=-1)

            ##########################################################
            # Calculate soft rounding targets and optimized threshold
            ##########################################################
            ptq_threshold_tensor_hat = torch.reshape(ptq_threshold_tensor, reshape_shape)

            #####################################################
            # Quantized Input
            #####################################################
            q_tensor = soft_rounding_symmetric_quantizer(input_tensor=inputs,
                                                         auxvar_tensor=aux_var,
                                                         threshold_tensor=ptq_threshold_tensor_hat,
                                                         num_bits=self.num_bits,
                                                         signed=True,
                                                         power_of_two=self.power_of_two)

            if self.quantization_parameter_learning and not self.power_of_two:
                scale = torch.reshape(self.get_quantizer_variable(SCALE_PTQ), reshape_shape)
                scale = torch.where(scale <= 0, torch.tensor(MIN_THRESHOLD, device=scale.device), scale)
                q_tensor *= scale

        else:
            q_tensor = soft_rounding_symmetric_quantizer(input_tensor=inputs,
                                                         auxvar_tensor=aux_var,
                                                         threshold_tensor=ptq_threshold_tensor,
                                                         num_bits=self.num_bits,
                                                         signed=True,
                                                         power_of_two=self.power_of_two)

            if self.quantization_parameter_learning and not self.power_of_two:
                scale = self.get_quantizer_variable(SCALE_PTQ)
                scale = torch.where(scale <= 0, torch.tensor(MIN_THRESHOLD, device=scale.device), scale)
                q_tensor *= scale

        return q_tensor
