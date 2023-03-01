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
from typing import List, Dict
import numpy as np

from model_compression_toolkit.core.common import Logger, max_power_of_two
from model_compression_toolkit import quantizers_infrastructure as qi
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.gptq.common.gptq_config import RoundingType
from model_compression_toolkit.gptq.pytorch.quantizer.base_pytorch_gptq_quantizer import \
    BasePytorchGPTQTrainableQuantizer
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, torch_tensor_to_numpy
from model_compression_toolkit.gptq.pytorch.quantizer import quant_utils as qutils
from model_compression_toolkit.gptq.common.gptq_constants import PTQ_THRESHOLD, SCALE_PTQ, N_EPOCHS, \
    MAX_ITERATIONS_DEFAULT, SOFT_ROUNDING_GAMMA, SOFT_ROUNDING_ZETA, SOFT_ROUNDING_BETA, GPTQ_ITER, AUXVAR
from model_compression_toolkit.core.common.constants import THRESHOLD, MIN_THRESHOLD
from model_compression_toolkit.quantizers_infrastructure import TrainableQuantizerWeightsConfig
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.common.base_inferable_quantizer import mark_quantizer
from model_compression_toolkit.quantizers_infrastructure.trainable_infrastructure.common.quant_utils import \
    get_threshold_reshape_shape


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


class LinearTempDecay:
    """
    Annealing process for the soft quantizer regularization temperature term.
    """

    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 20, end_b: int = 2):
        """
        Initializes a LinearTempDecay object.

        Args:
            t_max: maximal time step.
            rel_start_decay: Decay step size at the beginning of the process.
            start_b: Starting value of the regularization term.
            end_b: Target value of the regularization term.
        """

        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t: nn.Parameter) -> float:
        """
        Cosine annealing scheduler for soft quantizer regularization temperature term.

        Args:
            t: The current time step.

        Returns: Scheduled temperature.
        """

        is_before_start_decay = (t < self.start_decay).to(torch.float32)

        rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)

        return self.start_b * is_before_start_decay + \
               (1 - is_before_start_decay) * \
               (self.end_b + (self.start_b - self.end_b) * torch.maximum(to_torch_tensor(np.array([0.0])), (1 - rel_t)))


@mark_quantizer(quantization_target=qi.QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC],
                quantizer_type=RoundingType.SoftQuantizer)
class SymmetricSoftRoundingGPTQ(BasePytorchGPTQTrainableQuantizer):
    """
    Trainable symmetric quantizer to optimize the rounding of the quantized values using a soft quantization method.
    """

    def __init__(self,
                 quantization_config: TrainableQuantizerWeightsConfig,
                 n_batches: int = None,
                 quantization_parameter_learning: bool = False,
                 n_epochs: int = N_EPOCHS):
        """
        Construct a Pytorch model that utilize a fake weight quantizer of soft-quantizer for symmetric quantizer.

        Args:
            quantization_config: Trainable weights quantizer config.
            n_batches (int): number of batches in representative dataset
            quantization_parameter_learning (Bool): Whether to learn the threshold or not
            n_epochs (int): number of epochs the representative dataset is run during fine-tuning
        """

        if n_batches is None:
            Logger.error("SymmetricSoftRoundingGPTQ got an uninitialized n_batches argument.")

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
        # beta is used to set the regularization term.
        # See: https://arxiv.org/pdf/2004.10568.pdf
        self.gamma = SOFT_ROUNDING_GAMMA
        self.zeta = SOFT_ROUNDING_ZETA
        self.beta = SOFT_ROUNDING_BETA

        self.quantizer_parameters = {}

        # Initializing the temperature decay according to the number of expected gradient steps
        num_iterations = MAX_ITERATIONS_DEFAULT if n_batches is None else n_epochs * n_batches
        self.linear_decay = LinearTempDecay(num_iterations)

    def initialize_quantization(self,
                                tensor_shape: torch.Size,
                                name: str,
                                layer: qi.PytorchQuantizationWrapper) -> Dict[str, nn.Parameter]:
        """
        Return a dictionary of quantizer parameters and their names.

        Args:
            tensor_shape: tensor shape of the quantized tensor.
            name: Tensor name.
            layer: Layer to quantize.

        Returns:
            Dictionary of parameters names to the variables.
        """
        layer.register_parameter(f"{name}_{GPTQ_ITER}",
                                 nn.Parameter(to_torch_tensor(np.array([0])), requires_grad=False))

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
        self.quantizer_parameters = {PTQ_THRESHOLD: layer.get_parameter(f"{name}_{PTQ_THRESHOLD}"),
                                     AUXVAR: layer.get_parameter(f"{name}_{AUXVAR}"),
                                     GPTQ_ITER: layer.get_parameter(f"{name}_{GPTQ_ITER}")}

        if self.quantization_parameter_learning:
            layer.register_parameter(f"{name}_{SCALE_PTQ}",
                                     nn.Parameter(torch.ones_like(torch.Tensor(self.threshold_values)),
                                                  requires_grad=True))

            self.quantizer_parameters.update({SCALE_PTQ: layer.get_parameter(f"{name}_{SCALE_PTQ}")})

        return self.quantizer_parameters

    def get_regularization(self) -> torch.Tensor:
        """
        Computes the regularization term for the soft rounding loss.

        Returns:
            regularization term.
        """

        st = self.get_soft_targets()
        ar_iter = self.quantizer_parameters[GPTQ_ITER]
        b = self.linear_decay(ar_iter)
        return (1 - torch.pow(torch.abs(st - .5) * 2, b)).sum()

    def get_soft_targets(self) -> torch.Tensor:
        """
        Computes the rectified sigmoid function for the quantization target parameters.

        Returns:
            A tensor with the soft rounding targets values.

        """
        scaled_sigmoid = torch.sigmoid(self.quantizer_parameters[AUXVAR]) * (self.zeta - self.gamma) + self.gamma
        return torch.clip(scaled_sigmoid, min=0, max=1)

    def get_aux_variable(self) -> List[torch.Tensor]:
        """
        This function return a list with the quantizer's quantization auxiliary variables.

        Returns: A list with the quantization auxiliary variables.
        """
        return [self.quantizer_parameters.get(AUXVAR)]

    def get_quantization_variable(self) -> List[torch.Tensor]:
        """
        This function return a list with the quantizer's quantization parameters variables.

        Returns: A list with the quantization parameters.
        """
        if self.quantization_parameter_learning and not self.power_of_two:
            return [self.quantizer_parameters[SCALE_PTQ]]
        else:
            return []

    def get_quant_config(self) -> Dict[str, np.ndarray]:
        """
        Returns the config used to edit NodeQuantizationConfig after GPTQ retraining

        Returns:
            A dictionary of attributes the quantize_config retraining has changed during GPTQ retraining.
            Keys must match NodeQuantizationConfig attributes

        """
        old_threshold = torch_tensor_to_numpy(self.quantizer_parameters[PTQ_THRESHOLD])
        if self.power_of_two:
            old_threshold = max_power_of_two(old_threshold, MIN_THRESHOLD)
        else:
            if self.quantization_parameter_learning:
                scale = torch.reshape(self.quantizer_parameters[SCALE_PTQ], self.threshold_shape)
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
        ar_iter = self.quantizer_parameters[GPTQ_ITER]
        auxvar = self.quantizer_parameters[AUXVAR]
        ptq_threshold_tensor = self.quantizer_parameters[PTQ_THRESHOLD]

        #####################################################
        # Soft Rounding
        #####################################################
        aux_var = self.get_soft_targets()
        if training:
            ar_iter.set_(ar_iter + 1)
        else:
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
                scale = torch.reshape(self.quantizer_parameters[SCALE_PTQ], reshape_shape)
                q_tensor *= scale

        else:
            q_tensor = soft_rounding_symmetric_quantizer(input_tensor=inputs,
                                                         auxvar_tensor=aux_var,
                                                         threshold_tensor=ptq_threshold_tensor,
                                                         num_bits=self.num_bits,
                                                         signed=True,
                                                         power_of_two=self.power_of_two)

        return q_tensor
