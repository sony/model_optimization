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
from functools import partial
from typing import Callable

from model_compression_toolkit import GradientPTQConfig, RoundingType, GradientPTQConfigV2
from model_compression_toolkit.gptq.common.gptq_constants import N_EPOCHS
from model_compression_toolkit.gptq.keras.quantizer.soft_rounding.soft_quantizer_reg import \
    soft_quantizer_regularization


def get_regularization(gptq_config: GradientPTQConfig, representative_data_gen: Callable) -> Callable:
    if gptq_config.rounding_type == RoundingType.SoftQuantizer:
        # dry run on the representative dataset to count number of batches
        num_batches = 0
        for _ in representative_data_gen():
            num_batches += 1

        n_epochs = N_EPOCHS if not type(gptq_config) == GradientPTQConfigV2 else gptq_config.n_epochs
        return partial(soft_quantizer_regularization, n_batches=num_batches, n_epochs=n_epochs)
    else:
        return lambda m, e_reg: 0