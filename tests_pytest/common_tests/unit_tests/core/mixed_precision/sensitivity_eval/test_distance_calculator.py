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
from unittest.mock import Mock, patch

import numpy as np
import pytest

from model_compression_toolkit.core.common.framework_info import set_fw_info
from model_compression_toolkit.core import MixedPrecisionQuantizationConfig, MpDistanceWeighting
from model_compression_toolkit.core.common.hessian import HessianInfoService
from model_compression_toolkit.core.common.mixed_precision.sensitivity_eval.metric_calculators import \
    DistanceMetricCalculator
from model_compression_toolkit.logger import Logger


def repr_datagen():
    yield [np.random.rand(2, 16)]


class TestDistanceWeighting:
    ipts = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1]])
    out_pts = np.array([[1, 2], [3, 4], [5, 6]])

    @pytest.fixture
    def setup(self, mocker, graph_mock, fw_info_mock, fw_impl_mock):
        set_fw_info(fw_info_mock)
        mocker.patch.object(DistanceMetricCalculator, 'get_mp_interest_points', return_value=[None, None])
        mocker.patch.object(DistanceMetricCalculator, 'get_output_nodes_for_metric', return_value=[None])
        mocker.patch.object(DistanceMetricCalculator, '_init_metric_points_lists', return_value=(None, None))
        self.graph_mock = graph_mock
        fw_impl_mock.model_builder = Mock(return_value=(None, None))
        self.fw_impl_mock = fw_impl_mock

    @pytest.mark.parametrize('ipts, out_pts, method, exp_metric', [
        (ipts, np.array([]), MpDistanceWeighting.AVG, 0.55),
        (ipts, out_pts, MpDistanceWeighting.AVG, 0.55 + 3.5),
        (np.array([]), out_pts, MpDistanceWeighting.AVG, 3.5),
        (ipts, np.array([]), MpDistanceWeighting.LAST_LAYER, 0.95),
        (ipts, out_pts, MpDistanceWeighting.LAST_LAYER, 0.95 + 3.5),
        (np.array([]), out_pts, MpDistanceWeighting.LAST_LAYER, 3.5)
    ])
    def test_avg_and_last_layer(self, setup, ipts, out_pts, method, exp_metric):
        mp_cfg = MixedPrecisionQuantizationConfig(distance_weighting_method=method)
        self._run_test(ipts, out_pts, mp_cfg, exp_metric)

    @pytest.mark.parametrize('sigma', [0.1, 1.5])
    def test_exp(self, setup, sigma):
        mp_cfg = MixedPrecisionQuantizationConfig(distance_weighting_method=MpDistanceWeighting.EXP,
                                                  exp_distance_weighting_sigma=sigma)
        ipts = np.array([[0, 0],
                         [0.7 * sigma, 1.3 * sigma],
                         [np.log(5) * sigma, np.log(5) * sigma]])
        exp_ipts_metric = np.average(ipts.mean(1), weights=[0, 1-1/np.e, 0.8])

        self._run_test(ipts, np.array([]), mp_cfg, exp_ipts_metric)
        self._run_test(ipts, self.out_pts, mp_cfg, exp_ipts_metric + 3.5)

    def test_exp_zero_weights(self, setup, mocker):
        mp_cfg = MixedPrecisionQuantizationConfig(distance_weighting_method=MpDistanceWeighting.EXP,
                                                  exp_distance_weighting_sigma=1)
        ipts = np.array([[1e-9, 1e-8]], dtype=np.float32)
        warn_spy = mocker.patch.object(Logger, 'warning')
        self._run_test(ipts, np.array([]), mp_cfg, 0)
        assert 'All weights for interest points are 0.' in warn_spy.call_args.args[0]

    @pytest.mark.parametrize('via_flag', [True, False])
    def test_hessian_weights(self, setup, mocker, via_flag):
        hessians = np.random.rand(5)
        mocker.patch.object(DistanceMetricCalculator, '_compute_hessian_based_scores', return_value=hessians)
        if via_flag:
            mp_cfg = MixedPrecisionQuantizationConfig(use_hessian_based_scores=True)
        else:
            mp_cfg = MixedPrecisionQuantizationConfig(distance_weighting_method=MpDistanceWeighting.HESSIAN)
        kwargs = dict(hessian_info_service=Mock(spec=HessianInfoService))
        ipts_metric = np.average([.15, .35, .55, .75, .95], weights=hessians)
        self._run_test(self.ipts, np.array([]), mp_cfg, calc_kwargs=kwargs, exp_metric=ipts_metric)
        self._run_test(self.ipts, self.out_pts, mp_cfg, calc_kwargs=kwargs, exp_metric=ipts_metric + 3.5)

    def _run_test(self, ipts, out_pts, mp_cfg, exp_metric, calc_kwargs=None):
        mp_model = Mock()
        with patch.object(DistanceMetricCalculator, '_compute_distance', return_value=(ipts, out_pts)) as comp_dist_mock:
            calc = DistanceMetricCalculator(self.graph_mock, mp_cfg, repr_datagen, self.fw_impl_mock,
                                            **(calc_kwargs or {}))
            metric = calc.compute(mp_model)
        comp_dist_mock.assert_called_once_with(mp_model)
        assert np.allclose(metric, exp_metric)
