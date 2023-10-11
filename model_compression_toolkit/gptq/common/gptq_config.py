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
from typing import Callable, Any, Dict
from model_compression_toolkit.core.common.defaultdict import DefaultDict
from model_compression_toolkit.core import common
from model_compression_toolkit.gptq.common.gptq_constants import QUANT_PARAM_LEARNING_STR, MAX_LSB_STR, REG_DEFAULT


class RoundingType(Enum):
    """
    An enum for choosing the GPTQ rounding methods
    0. STRAIGHT-THROUGH ESTIMATOR
    1. SoftQuantizer
    """
    STE = 0
    SoftQuantizer = 1


class GPTQHessianWeightsConfig:
    """
    Configuration to use for computing the Hessian-based weights for GPTQ loss metric.
    """

    def __init__(self,
                 hessians_num_samples: int = 16,
                 norm_weights: bool = True,
                 log_norm: bool = True,
                 scale_log_norm: bool = False,
                 hessians_n_iter: int = 50): #TODO: remove

        """
        Initialize a GPTQHessianWeightsConfig.

        Args:
            hessians_num_samples (int): Number of samples to use for computing the Hessian-based weights.
            norm_weights (bool): Whether to normalize the returned weights (to get values between 0 and 1).
            log_norm (bool): Whether to use log normalization to the GPTQ Hessian-based weights.
            scale_log_norm (bool): Whether to scale the final vector of the Hessian weights.
            hessians_n_iter (int): Number of random iterations to run Hessian approximation for GPTQ weights.
        """

        self.hessians_num_samples = hessians_num_samples
        self.norm_weights = norm_weights
        self.log_norm = log_norm
        self.scale_log_norm = scale_log_norm
        self.hessians_n_iter = hessians_n_iter


class GradientPTQConfig:
    """
    Configuration to use for quantization with GradientPTQ (experimental).
    """

    def __init__(self, n_iter: int,
                 optimizer: Any,
                 optimizer_rest: Any = None,
                 loss: Callable = None,
                 log_function: Callable = None,
                 train_bias: bool = True,
                 rounding_type: RoundingType = RoundingType.SoftQuantizer,
                 use_hessian_based_weights: bool = True,
                 optimizer_quantization_parameter: Any = None,
                 optimizer_bias: Any = None,
                 regularization_factor: float = REG_DEFAULT,
                 hessian_weights_config: GPTQHessianWeightsConfig = GPTQHessianWeightsConfig(),
                 gptq_quantizer_params_override: Dict[str, Any] = None):
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
            rounding_type (RoundingType): An enum that defines the rounding type.
            use_hessian_based_weights (bool): Whether to use Hessian-based weights for weighted average loss.
            optimizer_quantization_parameter (Any): Optimizer to override the rest optimizer  for quantizer parameters.
            optimizer_bias (Any): Optimizer to override the rest optimizer for bias.
            regularization_factor (float): A floating point number that defines the regularization factor.
            hessian_weights_config (GPTQHessianWeightsConfig): A configuration that include all necessary arguments to run a computation of Hessian weights for the GPTQ loss.
            gptq_quantizer_params_override (dict): A dictionary of parameters to override in GPTQ quantizer instantiation. Defaults to None (no parameters).

        """
        self.n_iter = n_iter
        self.optimizer = optimizer
        self.optimizer_rest = optimizer_rest
        self.loss = loss
        self.log_function = log_function
        self.train_bias = train_bias

        self.rounding_type = rounding_type
        self.use_hessian_based_weights = use_hessian_based_weights
        self.optimizer_quantization_parameter = optimizer_quantization_parameter
        self.optimizer_bias = optimizer_bias
        self.regularization_factor = regularization_factor
        self.hessian_weights_config = hessian_weights_config

        self.gptq_quantizer_params_override = {} if gptq_quantizer_params_override is None \
            else gptq_quantizer_params_override


class GradientPTQConfigV2(GradientPTQConfig):
    """
    Configuration to use for quantization with GradientPTQV2 (experimental).
    """
    def __init__(self, n_epochs: int,
                 optimizer: Any,
                 optimizer_rest: Any = None,
                 loss: Callable = None,
                 log_function: Callable = None,
                 train_bias: bool = True,
                 rounding_type: RoundingType = RoundingType.SoftQuantizer,
                 use_hessian_based_weights: bool = True,
                 optimizer_quantization_parameter: Any = None,
                 optimizer_bias: Any = None,
                 regularization_factor: float = REG_DEFAULT,
                 hessian_weights_config: GPTQHessianWeightsConfig = GPTQHessianWeightsConfig(),
                 gptq_quantizer_params_override: Dict[str, Any] = None):
        """
        Initialize a GradientPTQConfigV2.

        Args:
            n_epochs (int): Number of representative dataset epochs to train.
            optimizer (Any): Optimizer to use.
            optimizer_rest (Any): Optimizer to use for bias and quantizer parameters.
            loss (Callable): The loss to use. should accept 6 lists of tensors. 1st list of quantized tensors, the 2nd list is the float tensors,
             the 3rd is a list of quantized weights, the 4th is a list of float weights, the 5th and 6th lists are the mean and std of the tensors
             accordingly. see example in multiple_tensors_mse_loss
            log_function (Callable): Function to log information about the GPTQ process.
            train_bias (bool): Whether to update the bias during the training or not.
            rounding_type (RoundingType): An enum that defines the rounding type.
            use_hessian_based_weights (bool): Whether to use Hessian-based weights for weighted average loss.
            optimizer_quantization_parameter (Any): Optimizer to override the rest optimizer  for quantizer parameters.
            optimizer_bias (Any): Optimizer to override the rest optimizerfor bias.
            regularization_factor (float): A floating point number that defines the regularization factor.
            hessian_weights_config (GPTQHessianWeightsConfig): A configuration that include all necessary arguments to run a computation of Hessian weights for the GPTQ loss.
            gptq_quantizer_params_override (dict): A dictionary of parameters to override in GPTQ quantizer instantiation. Defaults to None (no parameters).

        """

        super().__init__(n_iter=None,
                         optimizer=optimizer,
                         optimizer_rest=optimizer_rest,
                         loss=loss,
                         log_function=log_function,
                         train_bias=train_bias,
                         rounding_type=rounding_type,
                         use_hessian_based_weights=use_hessian_based_weights,
                         optimizer_quantization_parameter=optimizer_quantization_parameter,
                         optimizer_bias=optimizer_bias,
                         regularization_factor=regularization_factor,
                         hessian_weights_config=hessian_weights_config,
                         gptq_quantizer_params_override=gptq_quantizer_params_override)
        self.n_epochs = n_epochs

    @classmethod
    def from_v1(cls, n_ptq_iter: int, config_v1: GradientPTQConfig):
        """
        Initialize a GradientPTQConfigV2 from GradientPTQConfig instance.

        Args:
            n_ptq_iter (int): Number of PTQ calibration iters (length of representative dataset).
            config_v1 (GradientPTQConfig): A GPTQ config to convert to V2.

        """
        n_epochs = int(round(config_v1.n_iter) / n_ptq_iter)
        v1_params = config_v1.__dict__
        v1_params = {k: v for k, v in v1_params.items() if k != 'n_iter'}
        return cls(n_epochs, **v1_params)
