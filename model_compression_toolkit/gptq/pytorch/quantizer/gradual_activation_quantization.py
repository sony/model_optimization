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
from functools import partial
from typing import Callable

from model_compression_toolkit.gptq import GradientPTQConfig, QFractionLinearAnnealingConfig
from model_compression_toolkit.trainable_infrastructure import BasePytorchTrainableQuantizer

from model_compression_toolkit.trainable_infrastructure.pytorch.annealing_schedulers import LinearAnnealingScheduler


def get_gradual_activation_quantizer_wrapper_factory(gptq_config: GradientPTQConfig,
                                                     get_total_grad_steps_fn: Callable[[], int]) \
        -> Callable[[BasePytorchTrainableQuantizer], 'GradualActivationQuantizerWrapper']:
    """
    Get a factory for 'GradualActivationQuantizerWrapper'.

    Args:
        gptq_config: GPTQ configuration.
        get_total_grad_steps_fn: a callable to obtain the total expected number of gradient steps.

    Returns:
        A factory function to build 'GradualActivationQuantizerWrapper' from Quantizer.
    """
    if gptq_config.gradual_activation_quantization_config is None:
        return lambda q: q

    annealing_cfg = gptq_config.gradual_activation_quantization_config.q_fraction_scheduler_policy
    if isinstance(annealing_cfg, QFractionLinearAnnealingConfig):
        t_end = annealing_cfg.end_step or get_total_grad_steps_fn()
        factor_scheduler = LinearAnnealingScheduler(t_start=annealing_cfg.start_step, t_end=t_end,
                                                    initial_val=annealing_cfg.initial_q_fraction,
                                                    target_val=annealing_cfg.target_q_fraction)
    else:
        raise ValueError(f'Unknown annealing policy {annealing_cfg}')

    return partial(GradualActivationQuantizerWrapper, q_fraction_scheduler=factor_scheduler)


class GradualActivationQuantizerWrapper:
    # TODO update paper's url
    """
    Quantizer wrapper for Gradual Activation Quantization training (https://arxiv.org/abs/2309.11531).

    It computes the weighted sum of the float activation 'x' and the quantized activation 'q(x)':

      out = (1 - q_fraction) * x + q_fraction * q(x)

    where 'q_fraction' is a tensor fraction to quantize in the range [0, 1] provided by a scheduler.

    Args:
        quantizer: quantizer to wrap.
        q_fraction_scheduler: a callable that accepts a gradient step and returns the corresponding quantized fraction.
    """
    def __init__(self, quantizer: BasePytorchTrainableQuantizer, q_fraction_scheduler: Callable[[int], float]):
        self.quantizer = quantizer
        self.q_fraction_scheduler = q_fraction_scheduler
        self.step_cnt = 0

    def __call__(self, x, training: bool = True):
        q_fraction = self.q_fraction_scheduler(self.step_cnt)
        out_q = self.quantizer(x, training)
        out = (1 - q_fraction) * x + q_fraction * out_q
        self.step_cnt += 1
        return out

    def initialize_quantization(self, *args, **kwargs):
        self.quantizer.initialize_quantization(*args, **kwargs)
