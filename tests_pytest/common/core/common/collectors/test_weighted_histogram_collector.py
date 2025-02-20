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
from model_compression_toolkit.core.common.collectors.weighted_histogram_collector import WeightedHistogramCollector, \
    check_broadcastable
from model_compression_toolkit.logger import Logger


@pytest.fixture
def mock_logger():
    """Fixture to mock Logger.critical using unittest.mock.Mock."""
    mock = Mock()
    Logger.critical = mock  # Override Logger.critical with the mock
    return mock


@pytest.fixture
def collector():
    """Fixture that returns a WeightedHistogramCollector with a small number of bins for testing."""
    return WeightedHistogramCollector(n_bins=10)


class TestCheckBroadcastable:
    def test_valid_broadcast(self, mock_logger):
        """Test cases where broadcasting should succeed without calling Logger.critical."""

        # Same shape
        x = np.random.rand(4, 5, 6)
        w = np.random.rand(4, 5, 6)
        check_broadcastable(x, w)
        mock_logger.assert_not_called()

        # w has ones in dimensions
        w = np.random.rand(1, 5, 1)
        check_broadcastable(x, w)
        mock_logger.assert_not_called()

        # w has fewer dimensions but is still broadcastable
        w = np.random.rand(5, 6)
        check_broadcastable(x, w)
        mock_logger.assert_not_called()

        # w has only ones (fully broadcastable)
        w = np.random.rand(1, 1, 1)
        check_broadcastable(x, w)
        mock_logger.assert_not_called()

    def test_invalid_broadcast(self, mock_logger):
        """Test cases where broadcasting should fail and call Logger.critical."""

        x = np.random.rand(4, 5, 6)

        # More dimensions in w than x
        w = np.random.rand(4, 5, 6, 1)
        check_broadcastable(x, w)
        mock_logger.assert_called_once_with(
            f"Tensor weights with shape {w.shape} has more dimensions than tensor a with shape {x.shape}.")
        mock_logger.reset_mock()

        # Mismatched dimension (not 1 and not equal)
        w = np.random.rand(3, 5, 6)
        check_broadcastable(x, w)
        mock_logger.assert_called_once_with(
            f"Tensor weights with shape {w.shape} cannot be broadcasted to tensor a with shape {x.shape}. "
            f"Dimension mismatch at index 0: {w.shape[0]} cannot be broadcasted to {x.shape[0]}.")
        mock_logger.reset_mock()

        # Another mismatched case
        w = np.random.rand(4, 3, 6)
        check_broadcastable(x, w)
        mock_logger.assert_called_once_with(
            f"Tensor weights with shape {w.shape} cannot be broadcasted to tensor a with shape {x.shape}. "
            f"Dimension mismatch at index 1: {w.shape[1]} cannot be broadcasted to {x.shape[1]}.")
        mock_logger.reset_mock()

        # Another dimension mismatch
        w = np.random.rand(4, 5, 7)
        check_broadcastable(x, w)
        mock_logger.assert_called_once_with(
            f"Tensor weights with shape {w.shape} cannot be broadcasted to tensor a with shape {x.shape}. "
            f"Dimension mismatch at index 2: {w.shape[2]} cannot be broadcasted to {x.shape[2]}.")
        mock_logger.reset_mock()


class TestWeightedHistogramCollector:

    def test_update_uniform_weights(self, collector):
        # Update with no weights provided -> should use uniform weights.
        x = np.array([1, 2, 3, 4, 5])
        collector.update(x)
        bins, counts = collector.get_histogram()
        # Sum of counts should equal the number of samples.
        total_count = np.sum(counts)
        assert total_count == x.size, f"Expected total count {x.size}, got {total_count}"

    def test_update_with_weights(self, collector):
        # Update with provided weights.
        x = np.array([1, 2, 3, 4, 5])
        weights = np.array([0.5, 1.5, 2.0, 1.0, 0.0])
        collector.update(x, weights=weights)
        bins, counts = collector.get_histogram()
        total_weight = np.sum(counts)
        # Total weight in the merged histogram should be approximately the sum of weights.
        expected_weight = weights.sum()
        np.testing.assert_allclose(total_weight, expected_weight, err_msg="Total weight does not match expected sum.")

    def test_broadcasting_weights(self, collector):
        # Test with x having shape (2, 2) and weights of shape (2,)
        x = np.array([[1, 2],
                      [3, 4]])
        weights = np.array([0.5, 2.0])
        # weights shape is (2,) but x shape is (2,2). The update method should broadcast weights.
        collector.update(x, weights=weights)
        bins, counts = collector.get_histogram()
        total_weight = np.sum(counts)
        # After broadcasting, weights become:
        # [[0.5, 2.0],
        #  [0.5, 2.0]] so the expected sum is 0.5+2.0+0.5+2.0 = 5.0.
        np.testing.assert_allclose(total_weight, 5.0, err_msg="Broadcasting of weights did not work as expected.")

    def test_multiple_updates(self, collector):
        # Update the collector multiple times and check that the merged histogram total equals the sum over updates.
        x1 = np.array([1, 2, 3])
        weights1 = np.array([1, 2, 3])
        x2 = np.array([2, 3, 4])
        weights2 = np.array([0.5, 0.5, 0.5])
        collector.update(x1, weights=weights1)
        collector.update(x2, weights=weights2)
        bins, counts = collector.get_histogram()
        total_weight = np.sum(counts)
        expected_total = weights1.sum() + weights2.sum()
        np.testing.assert_allclose(total_weight, expected_total, err_msg="Total weight over multiple updates is incorrect.")

    def test_min_max(self, collector):
        # Update with data having a known min and max.
        x = np.linspace(0, 100, 50)
        collector.update(x)
        bins, counts = collector.get_histogram()
        # The binning is done automatically; we test that min() and max() return bin edges corresponding to nonzero counts.
        actual_min = collector.min()
        actual_max = collector.max()
        # Find the bins (except last one) that have nonzero counts.
        nonzero_bins = bins[:-1][counts > 0]
        expected_min = nonzero_bins.min()
        expected_max = nonzero_bins.max()
        assert actual_min == expected_min, "Minimum bin value does not match expected value."
        assert actual_max == expected_max, "Maximum bin value does not match expected value."

    def test_scale(self, collector):
        # Update collector with data, then scale with a single-channel factor.
        x = np.array([10, 20, 30])
        collector.update(x)
        bins_before, _ = collector.get_histogram()
        scale_factor = np.array([2.0])  # single-channel: valid scaling.
        collector.scale(scale_factor)
        bins_after, _ = collector.get_histogram()
        np.testing.assert_allclose(bins_after, bins_before * scale_factor, err_msg="Bins not scaled correctly.")

    def test_shift(self, collector):
        # Update collector with data, then shift with a single-channel shift.
        x = np.array([10, 20, 30])
        collector.update(x)
        bins_before, _ = collector.get_histogram()
        shift_value = np.array([5.0])
        collector.shift(shift_value)
        bins_after, _ = collector.get_histogram()
        np.testing.assert_allclose(bins_after, bins_before + shift_value, err_msg="Bins not shifted correctly.")

    def test_scale_per_channel_illegal(self, collector):
        # Test that providing a per-channel scale (multi-element array) marks the collector as illegal.
        x = np.array([1, 2, 3])
        collector.update(x)
        # Use a multi-element scale_factor to simulate per-channel scaling.
        scale_factor = np.array([2.0, 3.0])
        collector.scale(scale_factor)
        assert not collector.is_legal, "Collector should be marked illegal when scaling per-channel."

    def test_shift_per_channel_illegal(self, collector):
        # Test that providing a per-channel shift (multi-element array) marks the collector as illegal.
        x = np.array([1, 2, 3])
        collector.update(x)
        shift_value = np.array([1.0, 2.0])
        collector.shift(shift_value)
        assert not collector.is_legal, "Collector should be marked illegal when shifting per-channel."