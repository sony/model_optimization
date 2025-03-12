# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Iterable

from model_compression_toolkit.target_platform_capabilities import AttributeQuantizationConfig, \
    QuantizationConfigOptions
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR


def build_mp_config_options_for_kernel_bias_ops(base_w_config, base_op_config,
                                                w_nbits: Iterable[int], a_nbits: Iterable[int]):
    """
    Build mixed precision configuration for operators with kernel and bias (bias is not configurable).

    Args:
        base_w_config: base config for weights.
        base_op_config: base config for operator.
        w_nbits: bit configurations for weights.
        a_nbit: bit configurations for activation.

    Returns:

    """
    mp_configs = []
    bias_cfg = base_op_config.attr_weights_configs_mapping[BIAS_ATTR]
    for w_nbit in w_nbits:
        for a_nbit in a_nbits:
            attr_cfg = base_w_config.clone_and_edit(weights_n_bits=w_nbit)
            mp_configs.append(base_op_config.clone_and_edit(
                attr_weights_configs_mapping={KERNEL_ATTR: attr_cfg, BIAS_ATTR: bias_cfg},
                activation_n_bits=a_nbit
            ))
    return QuantizationConfigOptions(quantization_configurations=mp_configs, base_config=base_op_config)
