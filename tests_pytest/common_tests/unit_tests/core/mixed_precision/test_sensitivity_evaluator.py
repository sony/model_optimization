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
from unittest.mock import Mock

import numpy as np
import pytest
import types

from model_compression_toolkit.core import MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.common.mixed_precision.sensitivity_eval.metric_calculators import \
    DistanceMetricCalculator, CustomMetricCalculator
from model_compression_toolkit.core.common.mixed_precision.sensitivity_eval.sensitivity_evaluation import \
    SensitivityEvaluation


def repr_datagen():
    yield [np.random.rand(10)]


class TestSensitivityEvaluator:
    @pytest.mark.parametrize('custom', [True, False])
    def test_sensitivity_eval(self, fw_info_mock, fw_impl_mock, graph_mock, mocker, custom):
        """ This only tests correct calculator is created with correct args,
            and methods are called with correct args. Functional tests are at framework level. """
        build_mp_model_mock = mocker.patch.object(SensitivityEvaluation, '_build_mp_model', return_value=(Mock(), Mock()))
        configured_mock = mocker.patch.object(SensitivityEvaluation, '_configured_mp_model')

        calc_type = CustomMetricCalculator if custom else DistanceMetricCalculator

        # mock calculator init and compute
        def init(s, *args, **kwargs):
            s.all_interest_points = [1, 2, 3]
        init_spy = mocker.patch.object(calc_type, '__init__', side_effect=types.MethodType(init, calc_type))
        mocker.patch.object(calc_type, 'compute', return_value=42)

        kwargs = dict(custom_metric_fn=Mock()) if custom else {}
        mp_config = MixedPrecisionQuantizationConfig(**kwargs)
        hessian_mock = Mock()   # we only check the object is passed to calculator as is
        se = SensitivityEvaluation(graph_mock, mp_config, repr_datagen, fw_info=fw_info_mock, fw_impl=fw_impl_mock,
                                   hessian_info_service=hessian_mock)

        # compare exact types in case there is inheritance between calculators
        assert type(se.metric_calculator) is calc_type
        if custom:
            init_spy.assert_called_once_with(graph_mock, mp_config.custom_metric_fn)
        else:
            init_spy.assert_called_once_with(graph_mock, mp_config, repr_datagen, fw_info=fw_info_mock,
                                             fw_impl=fw_impl_mock, hessian_info_service=hessian_mock)

        build_mp_model_mock.assert_called_with(graph_mock, [1, 2, 3], False)
        assert se.mp_model == build_mp_model_mock.return_value[0]
        assert se.conf_node2layers == build_mp_model_mock.return_value[1]

        # check compute_metric
        mp_a_cfg = {'a': 1}
        mp_w_cfg = {'b': 2}
        assert se.compute_metric(mp_a_cfg=mp_a_cfg, mp_w_cfg=mp_w_cfg) == 42
        configured_mock.assert_called_once_with(mp_a_cfg, mp_w_cfg)
