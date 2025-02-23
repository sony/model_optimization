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
from mct_quantizers import QuantizationMethod
from unittest.mock import Mock

from pytest import fixture

from model_compression_toolkit.core import FrameworkInfo, QuantizationConfig
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.target_platform_capabilities import OpQuantizationConfig, Signedness, \
    QuantizationConfigOptions, TargetPlatformCapabilities


@fixture
def default_op_quant_cfg():
    return OpQuantizationConfig(
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


@fixture
def default_quant_cfg_options(default_op_quant_cfg):
    return QuantizationConfigOptions(quantization_configurations=[default_op_quant_cfg])


@fixture
def minimal_tpc(default_quant_cfg_options):
    return TargetPlatformCapabilities(default_qco=default_quant_cfg_options,
                                      tpc_platform_type='test',
                                      operator_set=None,
                                      fusing_patterns=None)


@fixture
def graph_mock():
    """ Basic Graph mock. """
    return Mock(spec_set=Graph, nodes=[])


@fixture
def fw_impl_mock():
    """ Basic FrameworkImplementation mock. """
    return Mock(spec_set=FrameworkImplementation)


@fixture
def fw_info_mock():
    """ Basic FrameworkInfo mock. """
    return Mock(spec_set=FrameworkInfo)
