# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Union, List
from abc import abstractmethod
import torch
import numpy as np
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfigV2
from model_compression_toolkit.core.common.logger import Logger
from model_compression_toolkit.gptq.pytorch.quantizer.gptq_quantizer import BaseWeightQuantizer
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig
from model_compression_toolkit.gptq.pytorch.quantizer.quant_utils import sample_gumbel
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.core.common.target_platform.op_quantization_config import QuantizationMethod
from model_compression_toolkit.gptq.pytorch.quantizer.quant_utils import ste_clip

P_INIT = 0.01
GR_SHIFT_BASE = 2


def init_aux_var(ceil_indicator: torch.Tensor, w_shape: torch.Size, m: int, p: float = P_INIT) -> torch.Tensor:
    """
    This function generate a random pi matrix for Gumbel Rounding
    Args:
        ceil_indicator: An array of indicator if the value should be ceil or floor.
        w_shape(torch.Size): A list of integers that represent the shape of the weights tensor to be quantization
        m(int):  An integer that define the number of shift
        p(float): A floating point number that represent the probability of non-round options of pi matrix.

    Returns: A torch tensor of pi tensor

    """

    if m < 2:
        Logger.error("m must be larger than two")
    if m % 2 != 0:
        Logger.error("m must be module two")
    m_hat = m // 2 - 1
    shift = -np.log(-np.log(1 - p))
    n = np.random.randn(*[m, *w_shape]) * np.sqrt(np.power(np.pi, 2) / 6)
    n = n.reshape([m, -1]).T
    ceil_indicator = ceil_indicator.cpu().numpy().flatten()
    n[np.arange(ceil_indicator.size), ceil_indicator + m_hat] += shift
    n = n.T.reshape(*[m, *w_shape])
    return torch.from_numpy(n).float()


def init_shift_var(m: int) -> torch.Tensor:
    """
    This function generate a tensor of 2*m+1 from -m to m
    Args:
        m: An integer value the represent m

    Returns: A tensor of size m

    """
    m_hat = m // 2
    aux_index_shift = [-m_hat + i + 1 for i in range(m)]
    return torch.Tensor(aux_index_shift)


class BaseGumbelWeightQuantizer(BaseWeightQuantizer):
    """
    Base class that implements a quantizer with trainable parameters to be used for GPTQ training.
    """

    def __init__(self,
                 weights_quantization_cfg: NodeWeightsQuantizationConfig,
                 gptq_config: GradientPTQConfigV2,
                 weight_shape: torch.Size):
        """
        Construct a Pytorch model that utilize a fake weight quantizer of Gumbel rounding
        Args:
            weights_quantization_cfg: Configuration of weight quantization
            gptq_config: GradientPTQConfigV2 object with parameters about the tuning process.
            weight_shape: weight shape for auxiliary tensor creation.
        """
        super().__init__()
        self.power_of_two = QuantizationMethod.POWER_OF_TWO == weights_quantization_cfg.weights_quantization_method
        self.reshape_aux_shift = [-1, *[1 for _ in range(len(weight_shape))]]
        self.num_bits = weights_quantization_cfg.weights_n_bits
        self.weight_shape = weight_shape
        self.max_delta_change = gptq_config.lsb_change_per_bit_width.get(self.num_bits)
        self.quantization_parameter_learning = gptq_config.quantization_parameters_learning
        self.m = GR_SHIFT_BASE * self.max_delta_change + GR_SHIFT_BASE
        self.minimal_temp = gptq_config.quantizer_config.minimal_temp
        self.maximal_temp = gptq_config.quantizer_config.maximal_temp
        self.temperature_learning = gptq_config.quantizer_config.temperature_learning
        self.cycle_iterations = max(1, int(gptq_config.n_epochs / gptq_config.quantizer_config.n_cycles))
        self.shift_tensor = to_torch_tensor(init_shift_var(self.m))
        self.tau = None
        self.g_t = 0
        self.p_t = None
        self.n_iter = 0
        self.update_gumbel_param = True
        scale = self.cycle_iterations / (-2 * np.log(0.001))

        self.gumbel_scale = gptq_config.quantizer_config.gumbel_scale
        self.gumbel_scale_per_bitwidth = gptq_config.quantizer_config.gumbel_scale_per_bitwidth

        def tau_function(i: int) -> float:
            """
            A function that generates the gumbel temperature.
            Args:
                i: An int that represents the current iteration number

            Returns: A temperature value.

            """
            if i < (self.cycle_iterations - 1):
                index = ((i + 1) % self.cycle_iterations) / scale
            else:
                index = (i % self.cycle_iterations) / scale

            x = np.exp(-index)
            return self.minimal_temp + (self.maximal_temp - self.minimal_temp) * x

        self.tau_function = tau_function

    def get_gumbel_probability(self) -> torch.Tensor:
        """
        A function that return the gumbel probability value.
        Returns: gumbel probability
        """
        return self.p_t

    def update_iteration(self, training):
        """
        A function that update parameters for GPTQ fine-tuning
        Args:
            training: whether in training mode or not
        """
        if self.temperature_learning:
            self.tau = ste_clip(self.temp_tensor, self.minimal_temp, self.maximal_temp)
        else:
            self.tau = self.tau_function(self.n_iter)
        if self.update_gumbel_param and training:
            self.n_iter += 1
            self.g_t = sample_gumbel([self.m, *self.weight_shape])

    @abstractmethod
    def get_temperature_variable(self) -> Union[torch.Tensor, List]:
        """
        Returns temperature trainable variables
        """
        raise Logger.error(f"{self.__class__.__name__} have to implement this abstract function.")