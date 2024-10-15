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
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any, Dict, Optional

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


@dataclass
class GPTQHessianScoresConfig:
    """
    Configuration to use for computing the Hessian-based scores for GPTQ loss metric.

    Args:
        hessians_num_samples (int|None): Number of samples to use for computing the Hessian-based scores.
          If None, compute Hessian for all images.
        norm_scores (bool): Whether to normalize the returned scores of the weighted loss function (to get values between 0 and 1).
        log_norm (bool): Whether to use log normalization for the GPTQ Hessian-based scores.
        scale_log_norm (bool): Whether to scale the final vector of the Hessian-based scores.
        hessian_batch_size (int): The Hessian computation batch size. used only if using GPTQ with Hessian-based objective.
        per_sample (bool): Whether to use per sample attention score.
    """
    hessians_num_samples: Optional[int] = GPTQ_HESSIAN_NUM_SAMPLES
    norm_scores: bool = True
    log_norm: bool = True
    scale_log_norm: bool = False
    hessian_batch_size: int = ACT_HESSIAN_DEFAULT_BATCH_SIZE
    per_sample: bool = False


@dataclass
class QFractionLinearAnnealingConfig:
    """
    Config for the quantized fraction linear scheduler of Gradual Activation Quantization.

    Args:
         initial_q_fraction: initial quantized fraction
         target_q_fraction: target quantized fraction
         start_step: gradient step to begin annealing
         end_step: gradient step to complete annealing. None means last step.
    """
    initial_q_fraction: float
    target_q_fraction: float
    start_step: int
    end_step: Optional[int]

    def __post_init__(self):
        if not (0 <= self.initial_q_fraction < self.target_q_fraction <= 1):
            raise ValueError(f'Expected 0 <= initial_q_fraction < target_q_fraction <= 1, received initial_q_fraction '
                             f'{self.initial_q_fraction} and target_q_fraction {self.target_q_fraction}.')
        if self.start_step < 0:
            raise ValueError(f'Expected start_step >= 0. received {self.start_step}.')
        if self.end_step is not None and self.end_step <= self.start_step:
            raise ValueError('Expected start_step < end_step, '
                             'received end_step {self.end_step} and start_step {self.start_stap}.')


@dataclass
class GradualActivationQuantizationConfig:
    """ Configuration for Gradual Activation Quantization.

        By default, the quantized fraction increases linearly from 0 to 1 throughout the training.

        Args:
            q_fraction_scheduler_policy: config for the scheduling of the quantized fraction.
                Only linear annealing is currently supported.
    """
    q_fraction_scheduler_policy: QFractionLinearAnnealingConfig = field(
        default_factory=lambda: QFractionLinearAnnealingConfig(initial_q_fraction=0,
                                                               target_q_fraction=1,
                                                               start_step=0,
                                                               end_step=None)
    )


@dataclass
class GradientPTQConfig:
    """
    Configuration to use for quantization with GradientPTQ.

    Args:
        n_epochs: Number of representative dataset epochs to train.
        optimizer: Optimizer to use.
        optimizer_rest: Optimizer to use for bias and quantizer parameters.
        loss: The loss to use. See 'multiple_tensors_mse_loss' for the expected interface.
        log_function: Function to log information about the GPTQ process.
        train_bias: Whether to update the bias during the training or not.
        rounding_type: An enum that defines the rounding type.
        use_hessian_based_weights: Whether to use Hessian-based weights for weighted average loss.
        optimizer_quantization_parameter: Optimizer to override the rest optimizer  for quantizer parameters.
        optimizer_bias: Optimizer to override the rest optimizer for bias.
        regularization_factor: A floating point number that defines the regularization factor.
        hessian_weights_config: A configuration that include all necessary arguments to run a computation of
            Hessian scores for the GPTQ loss.
        gradual_activation_quantization_config: A configuration for Gradual Activation Quantization.
        gptq_quantizer_params_override: A dictionary of parameters to override in GPTQ quantizer instantiation.
    """
    n_epochs: int
    optimizer: Any
    optimizer_rest: Any = None
    loss: Callable = None
    log_function: Callable = None
    train_bias: bool = True
    rounding_type: RoundingType = RoundingType.SoftQuantizer
    use_hessian_based_weights: bool = True
    optimizer_quantization_parameter: Any = None
    optimizer_bias: Any = None
    regularization_factor: float = REG_DEFAULT
    hessian_weights_config: GPTQHessianScoresConfig = field(default_factory=GPTQHessianScoresConfig)
    gradual_activation_quantization_config: Optional[GradualActivationQuantizationConfig] = None
    gptq_quantizer_params_override: Dict[str, Any] = field(default_factory=dict)
