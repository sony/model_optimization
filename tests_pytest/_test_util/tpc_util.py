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

from mct_quantizers import QuantizationMethod

from model_compression_toolkit.target_platform_capabilities import AttributeQuantizationConfig, \
    QuantizationConfigOptions, OpQuantizationConfig, Signedness, TargetPlatformCapabilities
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR


def build_mp_config_options_for_kernel_bias_ops(base_w_config: AttributeQuantizationConfig,
                                                base_op_config: OpQuantizationConfig,
                                                w_nbits: Iterable[int],
                                                a_nbits: Iterable[int]) -> QuantizationConfigOptions:
    """
    Build mixed precision configuration for operators with kernel and bias (bias is not configurable).

    Args:
        base_w_config: base attribute config for kernel.
        base_op_config: base config for operator.
        w_nbits: bit configurations for kernel.
        a_nbits: bit configurations for activation.

    Returns:
        QuantizationConfigOptions object.

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


def minimal_cfg_options():
    """
    Minimal op configuration options. Is intended to be used in integration tests,
    when real TPC (as opposed to mock) is needed, and we care about some of its content (like fusing)
    but we don't care about the default configuration options.
    """
    op_cfg = OpQuantizationConfig(
        default_weight_attr_config={},
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=[8],
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        simd_size=32,
        signedness=Signedness.AUTO)

    cfg_options = QuantizationConfigOptions(quantization_configurations=[op_cfg])

    return cfg_options


def minimal_tpc():
    """
    Minimal TPC. Is intended to be used in integration tests, when real TPC (as opposed to mock) is needed,
    but we don't care about its content.

    There is also a fixture form by the same name.
    """

    cfg_options = minimal_cfg_options()

    return TargetPlatformCapabilities(default_qco=cfg_options,
                                      tpc_platform_type='test',
                                      operator_set=None,
                                      fusing_patterns=None)
