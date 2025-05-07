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
import pytest
import numpy as np

from unittest.mock import Mock, patch, MagicMock

from model_compression_toolkit.core import MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.common.mixed_precision.sensitivity_evaluation import SensitivityEvaluation


def custom_float_metric(model_mp) -> float:
    return 100.0


def custom_np_float_metric(model_mp) -> np.floating:
    return np.float64(100.0)


def custom_str_metric(model_mp) -> str:
    return 'test'


def custom_none_metric(model_mp):
    return None


def get_sensitivity_evaluator(custom_metric_fn):
    mock_graph = Mock()
    mock_graph.get_topo_sorted_nodes.return_value = ['test', 'this', 'is', 'reset']
    mock_graph.get_outputs.return_value = []

    def representative_data_gen() -> list:
        for _ in range(5):
            yield np.random.randn(2, 3, 248, 248)

    mock_fw_info = Mock()

    def custom_to_tensor(img):
        return img

    mock_fw_impl = Mock()
    mock_fw_impl.to_tensor.side_effect = custom_to_tensor

    mp_cfg = MixedPrecisionQuantizationConfig(custom_metric_fn=custom_metric_fn)
    with patch.object(SensitivityEvaluation, '_build_models', return_value=(Mock(), Mock(), Mock())):
        sensitivity_eval = SensitivityEvaluation(graph=mock_graph,
                                                 quant_config=mp_cfg,
                                                 representative_data_gen=representative_data_gen,
                                                 fw_info=mock_fw_info,
                                                 fw_impl=mock_fw_impl)
    sensitivity_eval._configured_mp_model = MagicMock()
    return sensitivity_eval


class TestMPCustomMetricFunction:

    @pytest.mark.parametrize("metric_fn, expected", [
        (custom_float_metric, 100.0),
        (custom_np_float_metric, np.float64(100.0)),
    ])
    def test_valid_metric_function(self, metric_fn, expected):
        sensitivity_eval = get_sensitivity_evaluator(metric_fn)
        assert len(sensitivity_eval.interest_points) == 0
        assert sensitivity_eval.compute_metric({'test': 0}, {'test': 0}) == expected

    @pytest.mark.parametrize("metric_fn, expected", [
        (custom_str_metric, str.__name__),
        (custom_none_metric, type(None).__name__),
    ])
    def test_type_invalid_metric_function(self, metric_fn, expected):
        sensitivity_eval = get_sensitivity_evaluator(metric_fn)
        assert len(sensitivity_eval.interest_points) == 0
        with pytest.raises(TypeError, match=f'The custom_metric_fn is expected to return float or numpy float, got {expected}'):
            sensitivity_metric = sensitivity_eval.compute_metric(Mock(), Mock())
