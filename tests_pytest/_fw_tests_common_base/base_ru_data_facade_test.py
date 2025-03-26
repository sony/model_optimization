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
import abc
import copy
from typing import Callable

import numpy as np
import pytest

from model_compression_toolkit.core import CoreConfig, ResourceUtilization, QuantizationErrorMethod
from model_compression_toolkit.target_platform_capabilities import AttributeQuantizationConfig, OpQuantizationConfig, \
    Signedness, QuantizationConfigOptions, OperatorsSet, OperatorSetNames, TargetPlatformCapabilities, QuantizationMethod
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR
from tests_pytest._test_util.tpc_util import build_mp_config_options_for_kernel_bias_ops


def get_tpc():
    default_w_cfg = AttributeQuantizationConfig(weights_n_bits=4, enable_weights_quantization=True)
    default_op_cfg = OpQuantizationConfig(
        default_weight_attr_config=default_w_cfg,
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=[4, 8, 16],
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        simd_size=32,
        signedness=Signedness.AUTO)

    default_w_op_cfg = default_op_cfg.clone_and_edit(
        attr_weights_configs_mapping={KERNEL_ATTR: default_w_cfg, BIAS_ATTR: AttributeQuantizationConfig()}
    )
    mp_cfg_options = build_mp_config_options_for_kernel_bias_ops(base_w_config=default_w_cfg,
                                                                 base_op_config=default_w_op_cfg,
                                                                 w_nbits=[2, 4, 8], a_nbits=[4, 8, 16])

    linear_ops = [OperatorsSet(name=OperatorSetNames.CONV, qc_options=mp_cfg_options)]
    default_cfg = QuantizationConfigOptions(quantization_configurations=[default_op_cfg])
    tpc = TargetPlatformCapabilities(default_qco=default_cfg,
                                     tpc_platform_type='test',
                                     operator_set=linear_ops,
                                     fusing_patterns=None)
    return tpc


class BaseRUDataFacadeTest(abc.ABC):
    bhwc_input_shape = (1, 8, 12, 8)

    # api to test
    fw_ru_data_facade: Callable

    @abc.abstractmethod
    def _build_model(self, input_shape, out_chan, kernel, const):
        """ conv -> relu -> (add + const)
            Note: relu is added to prevent the folding of add. There is no fusion conv-relu as it is not defined in TPC.
        """
        raise NotImplementedError

    @pytest.mark.parametrize('error_method', [QuantizationErrorMethod.HMSE, QuantizationErrorMethod.MSE])
    def test_resource_utilization_data_facade(self, error_method):
        """ Integration test for resource data utilization user API.
            We include HMSE since it has special handling (not allowed without gptq, so changed to MSE,
            and make sure this change doesn't affect the original config. """
        tpc = get_tpc()

        core_cfg = CoreConfig()
        core_cfg.quantization_config.weights_error_method = error_method
        core_cfg_orig = copy.deepcopy(core_cfg)

        input_shape = (1, 8, 12, 8)
        model = self._build_model(input_shape=input_shape, out_chan=16, kernel=3, const=np.array([5]))

        def repr_data_gen():
            yield [np.random.random(input_shape)]

        facade = self.__class__.fw_ru_data_facade
        ru = facade(model, repr_data_gen, core_cfg, tpc)
        assert ru == ResourceUtilization(activation_memory=(16*10*6*2)*8/8,
                                         weights_memory=(16*8*3*3 + 1) * 4 / 8,
                                         total_memory=1920 + 576.5,
                                         bops=(8*16*3*3)*(10*6)*4*8)
        assert core_cfg.quantization_config.weights_error_method == error_method
        assert core_cfg == core_cfg_orig
