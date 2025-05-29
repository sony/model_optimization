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

from unittest.mock import Mock

from model_compression_toolkit.core.common.mixed_precision.sensitivity_eval.metric_calculators import \
    CustomMetricCalculator


def custom_metric_fn(model):
    return model.metric


class TestCustomMetricCalculator:
    @pytest.mark.parametrize('ret', [100.0, np.float64(100.0), np.float32(200.5)])
    def test_valid_metric_function(self, graph_mock, ret):
        graph_mock.get_outputs.return_value = [Mock(node=1), Mock(node=2)]
        calc = CustomMetricCalculator(graph_mock, custom_metric_fn=custom_metric_fn)
        assert calc.all_interest_points == [1, 2]
        assert calc.compute(Mock(metric=ret)) == ret

    @pytest.mark.parametrize('ret', ['foo', None])
    def test_invalid_metric_function(self, graph_mock, ret):
        graph_mock.get_outputs.return_value = [Mock(node=1), Mock(node=2)]
        calc = CustomMetricCalculator(graph_mock, custom_metric_fn=custom_metric_fn)
        with pytest.raises(TypeError, match=f'The custom_metric_fn is expected to return float or numpy float, '
                                            f'got {type(ret).__name__}'):
            calc.compute(Mock(metric=ret))

