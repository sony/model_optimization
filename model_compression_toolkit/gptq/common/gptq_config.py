# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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
from enum import Enum
from typing import Callable, Any
from model_compression_toolkit.core.common.defaultdict import DefaultDict
from model_compression_toolkit.core import common

MAX_LSBS_CHANGE_MAP = {8: 2,
                       4: 1,
                       2: 1}

N_CYCLES = 4
MIM_TEMP = 0.5
MAX_TEMP = 1.0


class RoundingType(Enum):
    """
    An enum for choosing the GPTQ rounding methods
    0. STRAIGHT-THROUGH ESTIMATOR
    1. Gumbel Rounding
    """
    STE = 0
    GumbelRounding = 1


class GumbelConfig(object):
    """
    Configuration to use for quantization with Gumbel Rounding.
    """

    def __init__(self,
                 temperature_learning: bool = True,
                 n_cycles: int = N_CYCLES,
                 minimal_temp: float = MIM_TEMP,
                 maximal_temp: float = MAX_TEMP,
                 gumbel_entropy_regularization: float = 0.01):
        """
        Initialize a GumbelConfig.


        Args:
            temperature_learning (bool): Whether to update the temperature during the training or not.
            gumbel_entropy_regularization (float): A floating point number that defines the gumbel entropy regularization factor.
            n_cycles (int): A floating point number that defines the gumbel entropy regularization factor.
            minimal_temp (float): A floating point number that defines the gumbel entropy regularization factor.
            maximal_temp (float): A floating point number that defines the gumbel entropy regularization factor.
        """
        self.gumbel_entropy_regularization = gumbel_entropy_regularization
        self.temperature_learning = temperature_learning
        self.n_cycles = n_cycles
        self.minimal_temp = minimal_temp
        self.maximal_temp = maximal_temp


class GradientPTQConfig:
    """
    Configuration to use for quantization with GradientPTQ (experimental).
    """

    def __init__(self,
                 n_iter: int,
                 optimizer: Any,
                 optimizer_rest: Any = None,
                 loss: Callable = None,
                 log_function: Callable = None,
                 train_bias: bool = True,
                 quantization_parameters_learning: bool = False,
                 sam_optimization: bool = False,
                 rounding_type: RoundingType = RoundingType.GumbelRounding,
                 rho: float = 0.01,
                 lsb_change_per_bit_width: dict = DefaultDict(MAX_LSBS_CHANGE_MAP, lambda: 1),
                 eps: float = 1e-6,
                 use_jac_based_weights: bool = True,
                 num_samples_for_loss: int = 16,
                 norm_weights: bool = False,
                 quantizer_config: GumbelConfig = GumbelConfig(),
                 optimizer_quantization_parameter: Any = None,
                 optimizer_bias: Any = None):
        """
        Initialize a GradientPTQConfig.

        Args:
            n_iter (int): Number of iterations to train.
            optimizer (Any): Optimizer to use.
            optimizer_rest (Any): Optimizer to use for bias and quantizer parameters.
            loss (Callable): The loss to use. should accept 6 lists of tensors. 1st list of quantized tensors, the 2nd list is the float tensors,
             the 3rd is a list of quantized weights, the 4th is a list of float weights, the 5th and 6th lists are the mean and std of the tensors
             accordingly. see example in multiple_tensors_mse_loss
            log_function (Callable): Function to log information about the GPTQ process.
            train_bias (bool): Whether to update the bias during the training or not.
            quantization_parameters_learning (bool): Whether to update the quantization param during the training or not.
            sam_optimization (bool): Whether to use sam optimization.
            rounding_type (RoundingType): An enum that defines the rounding type (STE or GumbelRoudning).
            rho (rho): A floating point number that defines the sam optimization lookahead.
            lsb_change_per_bit_width (dict): Whether to update the bias during the training or not.
            eps (float): A floating point value for numeric stability.
            use_jac_based_weights (bool): Whether to use jacobian-based weights for weighted average loss.
            num_samples_for_loss (int): Number of samples to use for computing the jacobian-based weights.
            norm_weights (bool): Whether to normalize the returned weights (to get values between 0 and 1).
            quantizer_config (Any): A class the contins the quantizer specific config.
            optimizer_quantization_parameter (Any): Optimizer to override the rest optimizer  for quantizer parameters.
            optimizer_bias (Any): Optimizer to override the rest optimizerfor bias.

        """
        self.n_iter = n_iter
        self.optimizer = optimizer
        self.optimizer_rest = optimizer_rest
        self.loss = loss
        self.log_function = log_function
        self.train_bias = train_bias
        self.quantization_parameters_learning = quantization_parameters_learning
        self.rounding_type = rounding_type
        self.sam_optimization = sam_optimization
        self.rho = rho
        self.lsb_change_per_bit_width = lsb_change_per_bit_width
        self.eps = eps
        self.use_jac_based_weights = use_jac_based_weights
        self.num_samples_for_loss = num_samples_for_loss
        self.norm_weights = norm_weights
        if not isinstance(quantizer_config, GumbelConfig) and self.is_gumbel:
            common.Logger.error("Please use GumbelConfig as quantizer config when using Gumbel Rounding")
        self.quantizer_config = quantizer_config
        self.optimizer_quantization_parameter = optimizer_quantization_parameter
        self.optimizer_bias = optimizer_bias

    @property
    def is_gumbel(self) -> bool:
        """
        This function state if Gumbel Rounding is in use.
        Returns: boolean

        """
        return self.rounding_type == RoundingType.GumbelRounding
