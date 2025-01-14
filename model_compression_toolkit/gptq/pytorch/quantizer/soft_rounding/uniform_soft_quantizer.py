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

from model_compression_toolkit.trainable_infrastructure.common.constants import FQ_MIN, FQ_MAX
from mct_quantizers import QuantizationMethod
from mct_quantizers import QuantizationTarget, PytorchQuantizationWrapper
from model_compression_toolkit.gptq.common.gptq_config import RoundingType
from model_compression_toolkit.gptq.pytorch.quantizer.base_pytorch_gptq_quantizer import \
    BasePytorchGPTQTrainableQuantizer
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, torch_tensor_to_numpy
from model_compression_toolkit.gptq.pytorch.quantizer import quant_utils as qutils
from model_compression_toolkit.gptq.common.gptq_constants import SOFT_ROUNDING_GAMMA, SOFT_ROUNDING_ZETA, AUXVAR
from model_compression_toolkit.gptq.pytorch.quantizer.quant_utils import fix_range_to_include_zero
from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerWeightsConfig
from mct_quantizers import mark_quantizer
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import \
    VariableGroup
from model_compression_toolkit.constants import RANGE_MAX, RANGE_MIN


def soft_rounding_unifrom_quantizer(input_tensor: torch.Tensor,
                                    auxvar_tensor: torch.Tensor,
                                    min_range: torch.Tensor,
                                    max_range: torch.Tensor,
                                    num_bits: int) -> torch.Tensor:
    """
    Quantize a tensor uniformly for GPTQ quantizers.

    Args:
        input_tensor: Tensor to quantize. values of this tensor are not changed during gptq.
        auxvar_tensor: Tensor that manifests the bit shift of the quantized weights due to gptq training.
        min_range: Tensor with min values to compute the delta grid.
        max_range: Tensor with max values to compute the delta grid.
        num_bits: Num of bits to use.

    Returns:
        A quantized tensor.
    """
    # adjusts the quantization range so the quantization grid includes zero.
    min_range, max_range = fix_range_to_include_zero(min_range, max_range, num_bits)
    delta = qutils.calculate_delta_uniform(min_range, max_range, num_bits)
    input_tensor_int = qutils.ste_floor((input_tensor - min_range) / delta)
    tensor_q = input_tensor_int + auxvar_tensor
    return delta * qutils.ste_clip(tensor_q,
                                   min_val=0,
                                   max_val=2 ** num_bits - 1) + min_range


@mark_quantizer(quantization_target=QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.UNIFORM],
                identifier=RoundingType.SoftQuantizer)
class UniformSoftRoundingGPTQ(BasePytorchGPTQTrainableQuantizer):
    """
    Trainable uniform quantizer to optimize the rounding of the quantized values using a soft quantization method.
    """

    def __init__(self,
                 quantization_config: TrainableQuantizerWeightsConfig,
                 quantization_parameter_learning: bool = False):
        """
        Construct a Pytorch model that utilize a fake weight quantizer of soft-quantizer for symmetric quantizer.

        Args:
            quantization_config: Trainable weights quantizer config.
            quantization_parameter_learning (Bool): Whether to learn the min/max ranges or not
        """

        super().__init__(quantization_config)
        self.num_bits = quantization_config.weights_n_bits
        self.per_channel = quantization_config.weights_per_channel_threshold

        self.min_values = quantization_config.weights_quantization_params[RANGE_MIN]
        self.max_values = quantization_config.weights_quantization_params[RANGE_MAX]

        self.quantization_axis = quantization_config.weights_channels_axis
        self.quantization_parameter_learning = quantization_parameter_learning

        # gamma and zeta are stretch parameters for computing the rectified sigmoid function.
        # See: https://arxiv.org/pdf/2004.10568.pdf
        self.gamma = SOFT_ROUNDING_GAMMA
        self.zeta = SOFT_ROUNDING_ZETA

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

        # Add min and max variables to layer.
        if self.per_channel:
            min_values = to_torch_tensor(self.min_values)
            max_values = to_torch_tensor(self.max_values)
        else:
            min_values = torch.tensor(self.min_values)
            max_values = torch.tensor(self.max_values)

        layer.register_parameter(name+"_"+FQ_MIN, nn.Parameter(min_values, requires_grad=self.quantization_parameter_learning))
        layer.register_parameter(name+"_"+FQ_MAX, nn.Parameter(max_values, requires_grad=self.quantization_parameter_learning))

        w = layer.layer.weight
        delta = qutils.calculate_delta_uniform(min_values, max_values, self.num_bits)
        w_clipped_normed = torch.clip((w - min_values) / delta, 0, 2 ** self.num_bits - 1)
        rest = w_clipped_normed - torch.floor(w_clipped_normed)  # rest of rounding [0, 1)
        alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
        layer.register_parameter(f"{name}_{AUXVAR}", nn.Parameter(alpha, requires_grad=True))

        # Save the quantizer parameters
        self.add_quantizer_variable(FQ_MIN, layer.get_parameter(name+"_"+FQ_MIN), VariableGroup.QPARAMS)
        self.add_quantizer_variable(FQ_MAX, layer.get_parameter(name+"_"+FQ_MAX), VariableGroup.QPARAMS)
        self.add_quantizer_variable(AUXVAR, layer.get_parameter(f"{name}_{AUXVAR}"), VariableGroup.WEIGHTS)

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
        min_values = torch_tensor_to_numpy(self.get_quantizer_variable(FQ_MIN))
        max_values = torch_tensor_to_numpy(self.get_quantizer_variable(FQ_MAX))
        return {RANGE_MIN:  min_values,
                RANGE_MAX:  max_values}

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
        min_range = self.get_quantizer_variable(FQ_MIN)
        max_range = self.get_quantizer_variable(FQ_MAX)

        #####################################################
        # Soft Rounding
        #####################################################
        aux_var = self.get_soft_targets()
        if not training:
            aux_var = (aux_var >= 0.5).to(auxvar.dtype)

        #####################################################
        # Quantized Input
        #####################################################
        q_tensor = soft_rounding_unifrom_quantizer(input_tensor=inputs,
                                                   auxvar_tensor=aux_var,
                                                   min_range=min_range,
                                                   max_range=max_range,
                                                   num_bits=self.num_bits)

        return q_tensor
