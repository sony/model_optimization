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

from model_compression_toolkit.constants import GPTQ_HESSIAN_NUM_SAMPLES, ACT_HESSIAN_DEFAULT_BATCH_SIZE
from model_compression_toolkit.gptq.common.gptq_constants import REG_DEFAULT


class RoundingType(Enum):
    """
    An enum for choosing the GPTQ rounding methods:

    STE - STRAIGHT-THROUGH ESTIMATOR

    SoftQuantizer - SoftQuantizer

    """
    STE = 0
    SoftQuantizer = 1


class GPTQHessianScoresConfig:
    """
    Configuration to use for computing the Hessian-based scores for GPTQ loss metric.
    """

    def __init__(self,
                 hessians_num_samples: int = GPTQ_HESSIAN_NUM_SAMPLES,
                 norm_scores: bool = True,
                 log_norm: bool = True,
                 scale_log_norm: bool = False,
                 hessian_batch_size: int = ACT_HESSIAN_DEFAULT_BATCH_SIZE):

        """
        Initialize a GPTQHessianWeightsConfig.

        Args:
            hessians_num_samples (int): Number of samples to use for computing the Hessian-based scores.
            norm_scores (bool): Whether to normalize the returned scores of the weighted loss function (to get values between 0 and 1).
            log_norm (bool): Whether to use log normalization for the GPTQ Hessian-based scores.
            scale_log_norm (bool): Whether to scale the final vector of the Hessian-based scores.
            hessian_batch_size (int): The Hessian computation batch size. used only if using GPTQ with Hessian-based objective.
        """

        self.hessians_num_samples = hessians_num_samples
        self.norm_scores = norm_scores
        self.log_norm = log_norm
        self.scale_log_norm = scale_log_norm
        self.hessian_batch_size = hessian_batch_size


class GradientPTQConfig:
    """
    Configuration to use for quantization with GradientPTQ.
    """
    def __init__(self,
                 n_epochs: int,
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
                 hessian_weights_config: GPTQHessianScoresConfig = GPTQHessianScoresConfig(),
                 gptq_quantizer_params_override: Dict[str, Any] = None):
        """
        Initialize a GradientPTQConfig.

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
            optimizer_bias (Any): Optimizer to override the rest optimizer for bias.
            regularization_factor (float): A floating point number that defines the regularization factor.
            hessian_weights_config (GPTQHessianScoresConfig): A configuration that include all necessary arguments to run a computation of Hessian scores for the GPTQ loss.
            gptq_quantizer_params_override (dict): A dictionary of parameters to override in GPTQ quantizer instantiation. Defaults to None (no parameters).

        """

        self.n_epochs = n_epochs
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


