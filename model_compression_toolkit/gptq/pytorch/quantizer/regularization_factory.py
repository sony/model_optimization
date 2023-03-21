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
from typing import Callable

from model_compression_toolkit.gptq import RoundingType, GradientPTQConfigV2, GradientPTQConfig
from model_compression_toolkit.gptq.pytorch.quantizer.soft_rounding.soft_quantizer_reg import \
    SoftQuantizerRegularization


def get_regularization(gptq_config: GradientPTQConfig, representative_data_gen: Callable) -> Callable:
    """
    Returns a function that computes the regularization term for GPTQ training based on the given
    rounding type in the GPTQ configuration.

    Args:
        gptq_config: A GPTQ configuration.
        representative_data_gen: Dataset used for the GPTQ training.

    Returns: A function for computing the regularization. If there is no regularization function defined for the given
        rounding type, then it returns a function that just returns 0.

    """
    if gptq_config.rounding_type == RoundingType.SoftQuantizer:
        # dry run on the representative dataset to count number of batches
        num_batches = 0
        for _ in representative_data_gen():
            num_batches += 1

        n_epochs = GradientPTQConfigV2.from_v1(n_ptq_iter=num_batches, config_v1=gptq_config).n_epochs if \
            not type(gptq_config) == GradientPTQConfigV2 else gptq_config.n_epochs
        return SoftQuantizerRegularization(total_gradient_steps=num_batches * n_epochs)
    else:
        return lambda m, e_reg: 0
