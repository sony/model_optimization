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

from tqdm import tqdm
from typing import Callable, Type

from model_compression_toolkit.gptq import RoundingType, GradientPTQConfig

# Common warmup fraction
WARMUP_STEP_FRACTION = 0.2


def get_regularization(gptq_config: GradientPTQConfig,
                       get_total_grad_steps_fn: Callable[[], int],
                       SoftQuantizerRegularizationFWClass: Type,
                       LinearAnnealingSchedulerFWClass: Type) -> Callable:
    """
    Returns a function that computes the regularization term for GPTQ training based on the given
    rounding type in the GPTQ configuration.

    Args:
        gptq_config: A GPTQ configuration.
        get_total_grad_steps_fn: a callable to obtain the total expected number of gradient steps.
        SoftQuantizerRegularizationFWClass: The class to use for soft quantizer regularization (framework-specific).
        LinearAnnealingSchedulerFWClass: The class to use for the annealing scheduler (framework-specific).

    Returns:
        Callable: A function for computing the regularization. If there is no regularization function
        defined for the given rounding type, then it returns a function that just returns 0.
    """
    if gptq_config.rounding_type == RoundingType.SoftQuantizer:
        total_gradient_steps = get_total_grad_steps_fn()
        t_start = int(WARMUP_STEP_FRACTION * total_gradient_steps)

        # Directly initializing the scheduler within the method
        scheduler = LinearAnnealingSchedulerFWClass(
            t_start=t_start,
            t_end=total_gradient_steps,
            initial_val=20,
            target_val=2
        )

        # Return the framework-specific soft quantizer regularization
        return SoftQuantizerRegularizationFWClass(scheduler)
    else:
        return lambda *args, **kwargs: 0
