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
import copy
from unittest.mock import Mock, MagicMock

import pytest

from model_compression_toolkit.core import CoreConfig, QuantizationErrorMethod, BitWidthConfig
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization_calculator import \
    ResourceUtilizationCalculator, TargetInclusionCriterion, BitwidthMode
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization_data import \
    compute_resource_utilization_data


class TestResourceUtilizationData:
    @pytest.mark.parametrize('error_method', [QuantizationErrorMethod.MSE, QuantizationErrorMethod.HMSE])
    def test_resource_utilization_data(self, fw_info_mock, fw_impl_mock, error_method, mocker):
        core_cfg = CoreConfig()
        core_cfg.quantization_config.weights_error_method = error_method
        core_cfg.bit_width_config = BitWidthConfig([1, 2])
        core_cfg_orig = copy.deepcopy(core_cfg)

        model_mock = Mock()
        data_gen_mock = Mock()
        fqc_mock = Mock()
        ru_calc_cls = mocker.patch('model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.'
                                   'resource_utilization_data.ResourceUtilizationCalculator',
                                   spec_set=ResourceUtilizationCalculator)
        ru_calc_cls.compute_resource_utilization = Mock()
        prep_runner = mocker.patch('model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.'
                                   'resource_utilization_data.graph_preparation_runner')

        compute_resource_utilization_data(model_mock,
                                          data_gen_mock,
                                          core_cfg,
                                          fqc_mock,
                                          fw_info_mock,
                                          fw_impl_mock)

        assert prep_runner.call_args.args == (model_mock,)
        passed_q_cfg = prep_runner.call_args.kwargs.pop('quantization_config')
        assert passed_q_cfg.weights_error_method == QuantizationErrorMethod.MSE
        assert prep_runner.call_args.kwargs == dict(representative_data_gen=data_gen_mock,
                                                    fw_info=fw_info_mock,
                                                    fw_impl=fw_impl_mock,
                                                    fqc=fqc_mock,
                                                    bit_width_config=core_cfg.bit_width_config,
                                                    mixed_precision_enable=False,
                                                    running_gptq=False)

        ru_calc_cls.assert_called_with(prep_runner.return_value, fw_info=fw_info_mock, fw_impl=fw_impl_mock)
        ru_calc_cls.return_value.compute_resource_utilization.assert_called_with(TargetInclusionCriterion.AnyQuantizedNonFused,
                                                                                 BitwidthMode.QDefaultSP)
        # make sure the original config wasn't changed
        assert core_cfg.quantization_config.weights_error_method == error_method
        assert core_cfg == core_cfg_orig

        # and wasn't used internally (its copy was)
        def assert_not_same_obj(a, b):
            assert type(a) is type(b) and a is not b
        assert_not_same_obj(core_cfg.bit_width_config, prep_runner.call_args.kwargs['bit_width_config'])
        assert_not_same_obj(core_cfg.quantization_config, passed_q_cfg)

