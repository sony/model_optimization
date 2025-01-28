# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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

import pytest
import numpy as np

from model_compression_toolkit.constants import MIN_THRESHOLD, THRESHOLD, SIGNED
from model_compression_toolkit.core import QuantizationErrorMethod
from model_compression_toolkit.core.common.quantization.quantization_params_generation.symmetric_selection import \
    symmetric_selection_histogram


@pytest.fixture
def hist():
    np.random.seed(42)
    size = (32, 32, 3)
    num_bins = 2048
    x = np.random.uniform(-7, 7, size=size).flatten()
    count, bins = np.histogram(x, bins=num_bins)

    return count, bins


@pytest.fixture
def bounded_hist():
    np.random.seed(42)
    size = (32, 32, 3)
    num_bins = 2048
    x = np.random.uniform(-7, 7, size=size).flatten()
    e_x = np.exp(x - np.max(x))
    x = (e_x / e_x.sum()) + 1
    count, bins = np.histogram(x, bins=num_bins)

    return count, bins


err_methods_to_test = [e.name for e in QuantizationErrorMethod if e != QuantizationErrorMethod.HMSE]


@pytest.mark.parametrize("error_method", err_methods_to_test)
def test_symmetric_threshold_selection(error_method, hist):
    counts, bins = hist

    search_res = symmetric_selection_histogram(bins, counts, 2, 8, Mock(), Mock(), Mock(), Mock(),
                                               MIN_THRESHOLD, QuantizationErrorMethod[error_method], True)

    assert THRESHOLD in search_res
    assert SIGNED in search_res
    assert np.isclose(search_res[THRESHOLD], 7, atol=0.4)
    assert search_res[SIGNED]


@pytest.mark.parametrize("error_method", err_methods_to_test)
def test_symmetric_threshold_selection_bounded_activation(error_method, bounded_hist):
    counts, bins = bounded_hist

    search_res = symmetric_selection_histogram(bins, counts, 2, 8, Mock(), Mock(), Mock(), Mock(),
                                               MIN_THRESHOLD, QuantizationErrorMethod[error_method], False)

    assert THRESHOLD in search_res
    assert SIGNED in search_res
    assert np.isclose(search_res[THRESHOLD], 1, atol=0.4)
    assert not search_res[SIGNED]
