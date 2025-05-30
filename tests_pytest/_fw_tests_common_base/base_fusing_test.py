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

from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    ResourceUtilization

from tests_pytest._test_util.fw_test_base import BaseFWIntegrationTest
import abc

import numpy as np

from model_compression_toolkit.core import CoreConfig
from tests_pytest._test_util.tpc_util import configure_mp_activation_opsets
from model_compression_toolkit.target_platform_capabilities import QuantizationMethod, AttributeQuantizationConfig, \
    OpQuantizationConfig, QuantizationConfigOptions, Signedness, OperatorSetNames, TargetPlatformCapabilities, Fusing, OperatorsSet


def build_activation_mp_tpc():
    default_op_cfg = OpQuantizationConfig(
        default_weight_attr_config=AttributeQuantizationConfig(),
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=[8],
        enable_activation_quantization=True,
        enable_weights_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        simd_size=32,
        signedness=Signedness.AUTO
    )

    opsets, _ = configure_mp_activation_opsets(
        opset_names=[OperatorSetNames.CONV,
                     OperatorSetNames.RELU,
                     OperatorSetNames.ADD,
                     OperatorSetNames.SIGMOID,
                     OperatorSetNames.FULLY_CONNECTED,
                     OperatorSetNames.HARDSWISH],
        base_op_config=default_op_cfg,
        a_nbits=[2, 4, 8]
    )

    sp_opsets, _ = configure_mp_activation_opsets(
        opset_names=[OperatorSetNames.SWISH],
        base_op_config=default_op_cfg,
        a_nbits=[8]
    )

    opsets.extend(sp_opsets)

    default_cfg = QuantizationConfigOptions(quantization_configurations=[default_op_cfg])

    tpc = TargetPlatformCapabilities(
        default_qco=default_cfg,
        tpc_platform_type='_test',
        operator_set=opsets,
        fusing_patterns=[
        Fusing(operator_groups=(
            OperatorsSet(name=OperatorSetNames.CONV),
            OperatorsSet(name=OperatorSetNames.RELU))),
        Fusing(operator_groups=(
            OperatorsSet(name=OperatorSetNames.CONV),
            OperatorsSet(name=OperatorSetNames.SWISH))),
        Fusing(operator_groups=(
            OperatorsSet(name=OperatorSetNames.CONV),
            OperatorsSet(name=OperatorSetNames.SIGMOID))),
        Fusing(operator_groups=(
            OperatorsSet(name=OperatorSetNames.FULLY_CONNECTED),
            OperatorsSet(name=OperatorSetNames.HARDSWISH))),

        ]
    )
    return tpc


class BaseFusingTest(BaseFWIntegrationTest, abc.ABC):

    bhwc_input_shape = None
    tpc = None
    fw_ptq_facade = None
    fw_ru_data_facade = None

    def _generate_repr_data(self):
        yield [np.random.random(self.bhwc_input_shape)]

    def _run_ptq(self, model, core_cfg=None, tpc=None, target_ru=None):
        if core_cfg is None:
            core_cfg = CoreConfig()
        if tpc is None:
            tpc = self.__class__.tpc
        return self.__class__.fw_ptq_facade(model,
                                            self._generate_repr_data,
                                            core_config=core_cfg,
                                            target_resource_utilization=target_ru,
                                            target_platform_capabilities=tpc)

    def _run_ru_facade(self, model, core_cfg=None, tpc=None):
        if core_cfg is None:
            core_cfg = CoreConfig()
        if tpc is None:
            tpc = self.__class__.tpc
        return self.__class__.fw_ru_data_facade(in_model=model,
                                                representative_data_gen=self._generate_repr_data,
                                                core_config=core_cfg,
                                                target_platform_capabilities=tpc)

    def _assert_quantizers_match(self, model_builder, expected_quantizers_fn=None, match_by_name=True):
        model = model_builder(self.bhwc_input_shape)
        quant_model, _ = self._run_ptq(model)
        actual_quantizers = self._get_actual_act_quant_holders(quant_model)

        if match_by_name and expected_quantizers_fn is not None:
            expected = expected_quantizers_fn()
            assert actual_quantizers == expected, f"Expected quantizers: {expected}, but got: {actual_quantizers}"
        else:
            expected_count = len(expected_quantizers_fn()) if expected_quantizers_fn else None
            assert isinstance(actual_quantizers, list)
            if expected_count is not None:
                assert len(actual_quantizers) == expected_count, (
                    f"Expected {expected_count} quantizers but got {len(actual_quantizers)}: {actual_quantizers}"
                )

    def test_quantized_model_contains_only_expected_activation_quantizers(self):
        self._assert_quantizers_match(self._build_test_model_basic_fusing,
                                      self._get_expected_act_quant_holders,
                                      match_by_name=False)

    def test_quantized_model_with_reuse_contains_only_expected_activation_quantizers(self):
        self._assert_quantizers_match(self._build_test_model_reuse,
                                      self._get_expected_act_quant_holders_in_reuse_model,
                                      match_by_name=False)

    def test_facade_ru_data_matches_expected_for_fused_graph(self):
        model = self._build_test_model_ru_data_facade(self.bhwc_input_shape)
        ru = self._run_ru_facade(model)
        assert isinstance(ru, ResourceUtilization)

        expected_ru = ResourceUtilization(
            weights_memory=3 * 3 * 1 * 1,
            activation_memory=18 * 18 * 3 * 3,
            total_memory=9 + 18 * 18 * 3 * 3,
            bops=3 * 1 * 1 * 18 * 18 * 3 * 8 * 8
        )
        assert ru == expected_ru

    def test_final_ru_data_is_correct(self):
        activation_memory_cr = 0.5
        tpc = build_activation_mp_tpc()
        self._assert_final_ru(self._build_test_model_ru_data_facade, activation_memory_cr, [2, 2], tpc=tpc)

    def test_facade_ru_data_matches_expected_with_snc_model(self):
        model = self._build_test_model_snc(self.bhwc_input_shape)
        ru = self._run_ru_facade(model)
        expected_activation_memory = 18 * 18 * 3 * 3
        assert ru.activation_memory == expected_activation_memory

    def test_final_ru_data_with_snc_model(self):
        for snc_enabled in [True, False]:
            activation_memory_cr = 0.75
            self._assert_final_ru(self._build_test_model_snc, activation_memory_cr,
                                  expected_mp_cfg=None, snc_enabled=snc_enabled, tpc=build_activation_mp_tpc())

    def _assert_final_ru(self, model_builder, activation_memory_cr, expected_mp_cfg=None, snc_enabled=True, tpc=None):
        core_cfg = CoreConfig()
        core_cfg.quantization_config.shift_negative_activation_correction = snc_enabled
        model = model_builder(self.bhwc_input_shape)

        target_ru = ResourceUtilization(
            activation_memory=(18 * 18 * 3 * 3) * activation_memory_cr
        )

        quant_model, ui = self._run_ptq(model, core_cfg=core_cfg, tpc=tpc, target_ru=target_ru)

        assert ui.final_resource_utilization.activation_memory == target_ru.activation_memory
        if expected_mp_cfg is not None:
            assert ui.mixed_precision_cfg == expected_mp_cfg

