# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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


import unittest
import numpy as np
from model_compression_toolkit.common.statistics_collector import StatsContainer
from model_compression_toolkit.common.statistics_collector import scale_statistics
from model_compression_toolkit.common.statistics_collector import shift_statistics


def init_stats_container(num_of_input_channels, init_min=None, init_max=None):
    np.random.seed(1)
    sc = StatsContainer(init_min_value=init_min, init_max_value=init_max)
    # by default stats collecter takes index -1 as index to
    # collect stats per-channel (when it's collected this way)
    x = np.random.rand(1, 2, 3, num_of_input_channels)
    for i in range(100):
        sc.update_statistics(x)
    return sc


def scale_stats_container(sc, num_of_scaling_factors):
    scaling_factor = np.random.random(num_of_scaling_factors)
    scaled_sc = scale_statistics(sc, scaling_factor)
    return scaled_sc, scaling_factor


def shift_stats_container(sc, num_of_shifting_factors):
    shifting_factor = np.random.random(num_of_shifting_factors)
    shifted_sc = shift_statistics(sc, shifting_factor)
    return shifted_sc, shifting_factor


class TestCollectorsManipulations(unittest.TestCase):

    ########### Test scaling ###########
    def test_mean_scale_per_channel(self, num_of_scaling_factors=10):
        sc = init_stats_container(num_of_scaling_factors)
        mean = sc.get_mean()
        scaled_sc, scaling_factor = scale_stats_container(sc, num_of_scaling_factors)
        scaled_mean = scaled_sc.get_mean()
        self.assertTrue(np.allclose(scaled_mean / scaling_factor, mean))

    def test_mean_scale_per_tensor(self, num_of_scaling_factors=1):
        sc = init_stats_container(num_of_scaling_factors)
        mean = sc.get_mean()
        scaled_sc, scaling_factor = scale_stats_container(sc, num_of_scaling_factors)
        scaled_mean = scaled_sc.get_mean()
        self.assertTrue(np.allclose(scaled_mean / scaling_factor, mean))

    def test_histogram_scale_per_channel(self, num_of_scaling_factors=10):
        sc = init_stats_container(num_of_scaling_factors)
        bins, _ = sc.hc.get_histogram()
        scaled_sc, scaling_factor = scale_stats_container(sc, num_of_scaling_factors)
        with self.assertRaises(Exception):
            scaled_sc.hc.get_histogram()  # data is corrupted. expect exception

    def test_histogram_scale_per_tensor(self, num_of_scaling_factors=1):
        sc = init_stats_container(num_of_scaling_factors)
        bins, _ = sc.hc.get_histogram()
        scaled_sc, scaling_factor = scale_stats_container(sc, num_of_scaling_factors)
        scaled_bins, _ = scaled_sc.hc.get_histogram()
        self.assertTrue(np.allclose(scaled_bins / scaling_factor, bins))

    def test_min_max_scale_per_channel(self, num_of_scaling_factors=10):
        sc = init_stats_container(num_of_scaling_factors)
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel
        scaled_sc, scaling_factor = scale_stats_container(sc, num_of_scaling_factors)
        min_pc_scaled, max_pc_scaled = scaled_sc.mpcc.min_per_channel, scaled_sc.mpcc.max_per_channel
        self.assertTrue(np.allclose(min_pc_scaled / scaling_factor, min_pc))
        self.assertTrue(np.allclose(max_pc_scaled / scaling_factor, max_pc))

    def test_min_max_scale_per_tensor(self, num_of_scaling_factors=1):
        sc = init_stats_container(num_of_scaling_factors)
        scaled_sc, scaling_factor = scale_stats_container(sc, num_of_scaling_factors)
        self.assertTrue(np.allclose(scaled_sc.get_min_max_values() / scaling_factor, sc.get_min_max_values()))

    ########### Test shifting ############
    def test_mean_shift_per_channel(self, num_of_shifting_factors=10):
        sc = init_stats_container(num_of_shifting_factors)
        mean = sc.get_mean()
        shifted_sc, shifting_factor = shift_stats_container(sc, num_of_shifting_factors)
        shifted_mean = shifted_sc.get_mean()
        self.assertTrue(np.allclose(shifted_mean - shifting_factor, mean))

    def test_mean_shift_per_tensor(self, num_of_shifting_factors=1):
        sc = init_stats_container(num_of_shifting_factors)
        mean = sc.get_mean()
        shifted_sc, shifting_factor = shift_stats_container(sc, num_of_shifting_factors)
        shifted_mean = shifted_sc.get_mean()
        self.assertTrue(np.allclose(shifted_mean - shifting_factor, mean))

    def test_histogram_shift_per_channel(self, num_of_shifting_factors=10):
        sc = init_stats_container(num_of_shifting_factors)
        bins, _ = sc.hc.get_histogram()
        shifted_sc, shifting_factor = shift_stats_container(sc, num_of_shifting_factors)
        with self.assertRaises(Exception):
            shifted_sc.hc.get_histogram()  # data is corrupted. expect exception

    def test_histogram_shift_per_tensor(self, num_of_shifting_factors=1):
        sc = init_stats_container(num_of_shifting_factors)
        bins, _ = sc.hc.get_histogram()
        shifted_sc, shifting_factor = shift_stats_container(sc, num_of_shifting_factors)
        shifted_bins, _ = shifted_sc.hc.get_histogram()
        self.assertTrue(np.allclose(shifted_bins - shifting_factor, bins))

    def test_min_max_shift_per_channel(self, num_of_shifting_factors=10):
        sc = init_stats_container(num_of_shifting_factors)
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel
        shifted_sc, shifting_factor = shift_stats_container(sc, num_of_shifting_factors)
        min_pc_shifted, max_pc_shifted = shifted_sc.mpcc.min_per_channel, shifted_sc.mpcc.max_per_channel
        self.assertTrue(np.allclose(min_pc_shifted - shifting_factor, min_pc))
        self.assertTrue(np.allclose(max_pc_shifted - shifting_factor, max_pc))

    def test_min_max_shift_per_tensor(self, num_of_shifting_factors=1):
        sc = init_stats_container(num_of_shifting_factors)
        shifted_sc, shifting_factor = shift_stats_container(sc, num_of_shifting_factors)
        self.assertTrue(np.allclose(shifted_sc.get_min_max_values() - shifting_factor, sc.get_min_max_values()))

    ########### Test scaling -> shifting (same granularity) ###########
    def test_mean_scale_shift_per_channel(self, num_of_input_channels=10):
        sc = init_stats_container(num_of_input_channels)
        mean = sc.get_mean()
        scaled_sc, scaling_factor = scale_stats_container(sc, num_of_input_channels)
        final_sc, shifting_factor = shift_stats_container(scaled_sc, num_of_input_channels)
        final_mean = final_sc.get_mean()
        restored_mean = (final_mean - shifting_factor) / scaling_factor
        self.assertTrue(np.allclose(restored_mean, mean))

    def test_mean_scale_shift_per_tensor(self, num_of_input_channels=1):
        sc = init_stats_container(num_of_input_channels)
        mean = sc.get_mean()
        scaled_sc, scaling_factor = scale_stats_container(sc, num_of_input_channels)
        final_sc, shifting_factor = shift_stats_container(scaled_sc, num_of_input_channels)
        final_mean = final_sc.get_mean()
        restored_mean = (final_mean - shifting_factor) / scaling_factor
        self.assertTrue(np.allclose(restored_mean, mean))

    def test_histogram_scale_shift_per_channel(self, num_of_input_channels=10):
        sc = init_stats_container(num_of_input_channels)
        bins, _ = sc.hc.get_histogram()
        scaled_sc, scaling_factor = scale_stats_container(sc, num_of_input_channels)
        final_sc, shifting_factor = shift_stats_container(scaled_sc, num_of_input_channels)
        with self.assertRaises(Exception):
            final_sc.hc.get_histogram()

    def test_histogram_scale_shift_per_tensor(self, num_of_input_channels=1):
        sc = init_stats_container(num_of_input_channels)
        bins, _ = sc.hc.get_histogram()
        scaled_sc, scaling_factor = scale_stats_container(sc, num_of_input_channels)
        final_sc, shifting_factor = shift_stats_container(scaled_sc, num_of_input_channels)
        final_bins, _ = final_sc.hc.get_histogram()
        restored_bins = (final_bins - shifting_factor) / scaling_factor
        self.assertTrue(np.allclose(restored_bins, bins))

    def test_minmax_scale_shift_per_channel(self, num_of_input_channels=10):
        sc = init_stats_container(num_of_input_channels)
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel
        scaled_sc, scaling_factor = scale_stats_container(sc, num_of_input_channels)
        final_sc, shifting_factor = shift_stats_container(scaled_sc, num_of_input_channels)
        min_pc_final, max_pc_final = final_sc.mpcc.min_per_channel, final_sc.mpcc.max_per_channel
        restored_min_pc = (min_pc_final - shifting_factor) / scaling_factor
        restored_max_pc = (max_pc_final - shifting_factor) / scaling_factor
        self.assertTrue(np.allclose(restored_min_pc, min_pc))
        self.assertTrue(np.allclose(restored_max_pc, max_pc))

    def test_minmax_scale_shift_per_tensor(self, num_of_input_channels=1):
        sc = init_stats_container(num_of_input_channels)
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel
        scaled_sc, scaling_factor = scale_stats_container(sc, num_of_input_channels)
        final_sc, shifting_factor = shift_stats_container(scaled_sc, num_of_input_channels)
        min_pc_final, max_pc_final = final_sc.mpcc.min_per_channel, final_sc.mpcc.max_per_channel
        restored_min_pc = (min_pc_final - shifting_factor) / scaling_factor
        restored_max_pc = (max_pc_final - shifting_factor) / scaling_factor
        self.assertTrue(np.allclose(restored_min_pc, min_pc))
        self.assertTrue(np.allclose(restored_max_pc, max_pc))

    ########### Test scaling -> shifting -> scaling (same granularity) ###########
    def test_mean_scale_shift_scale_per_channel(self, num_of_input_channels=10):
        sc = init_stats_container(num_of_input_channels)
        mean = sc.get_mean()

        scaled_sc, scaling_factor = scale_stats_container(sc, num_of_input_channels)
        shifted_sc, shifting_factor = shift_stats_container(scaled_sc, num_of_input_channels)
        final_sc, scaling_factor2 = scale_stats_container(shifted_sc, num_of_input_channels)

        final_mean = final_sc.get_mean()
        restored_mean = (final_mean / scaling_factor2 - shifting_factor) / scaling_factor
        self.assertTrue(np.allclose(restored_mean, mean))

    def test_mean_scale_shift_scale_per_tensor(self, num_of_input_channels=1):
        sc = init_stats_container(num_of_input_channels)
        mean = sc.get_mean()

        scaled_sc, scaling_factor = scale_stats_container(sc, num_of_input_channels)
        shifted_sc, shifting_factor = shift_stats_container(scaled_sc, num_of_input_channels)
        final_sc, scaling_factor2 = scale_stats_container(shifted_sc, num_of_input_channels)

        final_mean = final_sc.get_mean()
        restored_mean = (final_mean / scaling_factor2 - shifting_factor) / scaling_factor
        self.assertTrue(np.allclose(restored_mean, mean))

    def test_histogram_scale_shift_scale_per_channel(self, num_of_input_channels=10):
        sc = init_stats_container(num_of_input_channels)
        bins, _ = sc.hc.get_histogram()

        scaled_sc, scaling_factor = scale_stats_container(sc, num_of_input_channels)
        shifted_sc, shifting_factor = shift_stats_container(scaled_sc, num_of_input_channels)
        final_sc, scaling_factor2 = scale_stats_container(shifted_sc, num_of_input_channels)

        with self.assertRaises(Exception):
            final_sc.hc.get_histogram()

    def test_histogram_scale_shift_scale_per_tensor(self, num_of_input_channels=1):
        sc = init_stats_container(num_of_input_channels)
        bins, _ = sc.hc.get_histogram()

        scaled_sc, scaling_factor = scale_stats_container(sc, num_of_input_channels)
        shifted_sc, shifting_factor = shift_stats_container(scaled_sc, num_of_input_channels)
        final_sc, scaling_factor2 = scale_stats_container(shifted_sc, num_of_input_channels)

        final_bins, _ = final_sc.hc.get_histogram()
        restored_bins = (final_bins / scaling_factor2 - shifting_factor) / scaling_factor
        self.assertTrue(np.allclose(restored_bins, bins))

    def test_minmax_scale_shift_scale_per_channel(self, num_of_input_channels=10):
        sc = init_stats_container(num_of_input_channels)
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel

        scaled_sc, scaling_factor = scale_stats_container(sc, num_of_input_channels)
        shifted_sc, shifting_factor = shift_stats_container(scaled_sc, num_of_input_channels)
        final_sc, scaling_factor2 = scale_stats_container(shifted_sc, num_of_input_channels)

        min_pc_final, max_pc_final = final_sc.mpcc.min_per_channel, final_sc.mpcc.max_per_channel
        restored_min_pc = (min_pc_final / scaling_factor2 - shifting_factor) / scaling_factor
        restored_max_pc = (max_pc_final / scaling_factor2 - shifting_factor) / scaling_factor

        self.assertTrue(np.allclose(restored_min_pc, min_pc))
        self.assertTrue(np.allclose(restored_max_pc, max_pc))

    def test_minmax_scale_shift_scale_per_tensor(self, num_of_input_channels=1):
        sc = init_stats_container(num_of_input_channels)
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel

        scaled_sc, scaling_factor = scale_stats_container(sc, num_of_input_channels)
        shifted_sc, shifting_factor = shift_stats_container(scaled_sc, num_of_input_channels)
        final_sc, scaling_factor2 = scale_stats_container(shifted_sc, num_of_input_channels)

        min_pc_final, max_pc_final = final_sc.mpcc.min_per_channel, final_sc.mpcc.max_per_channel
        restored_min_pc = (min_pc_final / scaling_factor2 - shifting_factor) / scaling_factor
        restored_max_pc = (max_pc_final / scaling_factor2 - shifting_factor) / scaling_factor
        self.assertTrue(np.allclose(restored_min_pc, min_pc))
        self.assertTrue(np.allclose(restored_max_pc, max_pc))

    ########### Test scaling -> shifting (different granularity) ###########
    def test_mean_scale_per_channel_shift_per_tensor(self, num_scale_factors=10, num_shift_factors=1):
        sc = init_stats_container(max(num_scale_factors, num_shift_factors))
        mean = sc.get_mean()

        scaled_sc, scaling_factor = scale_stats_container(sc, num_scale_factors)
        final_sc, shifting_factor = shift_stats_container(scaled_sc, num_shift_factors)

        final_mean = final_sc.get_mean()
        restored_mean = (final_mean - shifting_factor) / scaling_factor
        self.assertTrue(np.allclose(restored_mean, mean))

    def test_mean_scale_per_tensor_shift_per_channel(self, num_scale_factors=1, num_shift_factors=10):
        sc = init_stats_container(max(num_scale_factors, num_shift_factors))
        mean = sc.get_mean()

        scaled_sc, scaling_factor = scale_stats_container(sc, num_scale_factors)
        final_sc, shifting_factor = shift_stats_container(scaled_sc, num_shift_factors)

        final_mean = final_sc.get_mean()
        restored_mean = (final_mean - shifting_factor) / scaling_factor
        self.assertTrue(np.allclose(restored_mean, mean))

    def test_histogram_scale_per_channel_shift_per_tensor(self, num_scale_factors=10, num_shift_factors=1):
        sc = init_stats_container(max(num_scale_factors, num_shift_factors))

        scaled_sc, scaling_factor = scale_stats_container(sc, num_scale_factors)
        final_sc, shifting_factor = shift_stats_container(scaled_sc, num_shift_factors)

        with self.assertRaises(Exception):
            final_sc.hc.get_histogram()

    def test_histogram_scale_per_tensor_shift_per_channel(self, num_scale_factors=1, num_shift_factors=10):
        sc = init_stats_container(max(num_scale_factors, num_shift_factors))

        scaled_sc, scaling_factor = scale_stats_container(sc, num_scale_factors)
        final_sc, shifting_factor = shift_stats_container(scaled_sc, num_shift_factors)

        with self.assertRaises(Exception):
            final_sc.hc.get_histogram()

    def test_minmax_scale_per_channel_shift_per_tensor(self, num_scale_factors=10, num_shift_factors=1):
        sc = init_stats_container(max(num_scale_factors, num_shift_factors))
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel

        scaled_sc, scaling_factor = scale_stats_container(sc, num_scale_factors)
        final_sc, shifting_factor = shift_stats_container(scaled_sc, num_shift_factors)

        min_pc_final, max_pc_final = final_sc.mpcc.min_per_channel, final_sc.mpcc.max_per_channel

        restored_min_pc = (min_pc_final - shifting_factor) / scaling_factor
        restored_max_pc = (max_pc_final - shifting_factor) / scaling_factor
        self.assertTrue(np.allclose(restored_min_pc, min_pc))
        self.assertTrue(np.allclose(restored_max_pc, max_pc))

    def test_minmax_scale_per_tensor_shift_per_channel(self, num_scale_factors=1, num_shift_factors=10):
        sc = init_stats_container(max(num_scale_factors, num_shift_factors))
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel

        scaled_sc, scaling_factor = scale_stats_container(sc, num_scale_factors)
        final_sc, shifting_factor = shift_stats_container(scaled_sc, num_shift_factors)

        min_pc_final, max_pc_final = final_sc.mpcc.min_per_channel, final_sc.mpcc.max_per_channel

        restored_min_pc = (min_pc_final - shifting_factor) / scaling_factor
        restored_max_pc = (max_pc_final - shifting_factor) / scaling_factor
        self.assertTrue(np.allclose(restored_min_pc, min_pc))
        self.assertTrue(np.allclose(restored_max_pc, max_pc))

    ########### Test shifting -> scaling (different granularity) ###########

    def test_mean_shift_per_channel_scale_per_tensor(self, num_scale_factors=10, num_shift_factors=1):
        sc = init_stats_container(max(num_scale_factors, num_shift_factors))
        mean = sc.get_mean()

        shifted_sc, shifting_factor = shift_stats_container(sc, num_shift_factors)
        final_sc, scaling_factor = scale_stats_container(shifted_sc, num_scale_factors)

        final_mean = final_sc.get_mean()
        restored_mean = final_mean / scaling_factor - shifting_factor
        self.assertTrue(np.allclose(restored_mean, mean))

    def test_mean_shift_per_tensor_scale_per_channel(self, num_scale_factors=1, num_shift_factors=10):
        sc = init_stats_container(max(num_scale_factors, num_shift_factors))
        mean = sc.get_mean()

        shifted_sc, shifting_factor = shift_stats_container(sc, num_shift_factors)
        final_sc, scaling_factor = scale_stats_container(shifted_sc, num_scale_factors)

        final_mean = final_sc.get_mean()
        restored_mean = final_mean / scaling_factor - shifting_factor
        self.assertTrue(np.allclose(restored_mean, mean))

    def test_histogram_shift_per_channel_scale_per_tensor(self, num_scale_factors=10, num_shift_factors=1):
        sc = init_stats_container(max(num_scale_factors, num_shift_factors))

        shifted_sc, shifting_factor = shift_stats_container(sc, num_shift_factors)
        final_sc, scaling_factor = scale_stats_container(shifted_sc, num_scale_factors)

        with self.assertRaises(Exception):
            final_sc.hc.get_histogram()

    def test_histogram_shift_per_tensor_scale_per_channel(self, num_scale_factors=1, num_shift_factors=10):
        sc = init_stats_container(max(num_scale_factors, num_shift_factors))

        shifted_sc, shifting_factor = shift_stats_container(sc, num_shift_factors)
        final_sc, scaling_factor = scale_stats_container(shifted_sc, num_scale_factors)

        with self.assertRaises(Exception):
            final_sc.hc.get_histogram()

    def test_minmax_shift_per_channel_scale_per_tensor(self, num_scale_factors=10, num_shift_factors=1):
        sc = init_stats_container(max(num_scale_factors, num_shift_factors))
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel

        shifted_sc, shifting_factor = shift_stats_container(sc, num_shift_factors)
        final_sc, scaling_factor = scale_stats_container(shifted_sc, num_scale_factors)

        min_pc_final, max_pc_final = final_sc.mpcc.min_per_channel, final_sc.mpcc.max_per_channel

        restored_min_pc = min_pc_final / scaling_factor - shifting_factor
        restored_max_pc = max_pc_final / scaling_factor - shifting_factor
        self.assertTrue(np.allclose(restored_min_pc, min_pc))
        self.assertTrue(np.allclose(restored_max_pc, max_pc))

    def test_minmax_shift_per_tensor_scale_per_channel(self, num_scale_factors=1, num_shift_factors=10):
        sc = init_stats_container(max(num_scale_factors, num_shift_factors))
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel

        shifted_sc, shifting_factor = shift_stats_container(sc, num_shift_factors)
        final_sc, scaling_factor = scale_stats_container(shifted_sc, num_scale_factors)

        min_pc_final, max_pc_final = final_sc.mpcc.min_per_channel, final_sc.mpcc.max_per_channel

        restored_min_pc = min_pc_final / scaling_factor - shifting_factor
        restored_max_pc = max_pc_final / scaling_factor - shifting_factor
        self.assertTrue(np.allclose(restored_min_pc, min_pc))
        self.assertTrue(np.allclose(restored_max_pc, max_pc))

    ########### Test shifting for collector with init values ###########
    def test_minmax_shift_per_channel_init_min(self, num_of_shifting_factors=10):
        sc = init_stats_container(num_of_shifting_factors, init_min=0)
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel
        shifted_sc, shifting_factor = shift_stats_container(sc, num_of_shifting_factors)
        min_pc_final, max_pc_final = shifted_sc.mpcc.min_per_channel, shifted_sc.mpcc.max_per_channel
        restored_min_pc = min_pc_final - shifting_factor
        restored_max_pc = max_pc_final - shifting_factor
        self.assertTrue(np.allclose(restored_min_pc, min_pc))
        self.assertTrue(np.allclose(restored_max_pc, max_pc))

        min_final, max_final = shifted_sc.get_min_max_values()
        self.assertTrue(np.allclose(min_final, np.min(min_pc + shifting_factor)))
        self.assertTrue(np.allclose(max_final, np.max(max_pc + shifting_factor)))

    def test_minmax_shift_per_channel_init_max(self, num_of_shifting_factors=10):
        sc = init_stats_container(num_of_shifting_factors, init_max=99)
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel
        shifted_sc, shifting_factor = shift_stats_container(sc, num_of_shifting_factors)
        min_pc_final, max_pc_final = shifted_sc.mpcc.min_per_channel, shifted_sc.mpcc.max_per_channel
        restored_min_pc = min_pc_final - shifting_factor
        restored_max_pc = max_pc_final - shifting_factor
        self.assertTrue(np.allclose(restored_min_pc, min_pc))
        self.assertTrue(np.allclose(restored_max_pc, max_pc))

        min_final, max_final = shifted_sc.get_min_max_values()
        # init values should be ignored as stats were shifted per-channel
        self.assertTrue(np.allclose(max_final, np.max(max_pc + shifting_factor)))
        self.assertTrue(np.allclose(min_final, np.min(min_pc + shifting_factor)))

    def test_minmax_shift_per_tensor_init_min(self, num_of_shifting_factors=1):
        sc = init_stats_container(num_of_shifting_factors, init_min=0)
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel
        shifted_sc, shifting_factor = shift_stats_container(sc, num_of_shifting_factors)
        min_pc_final, max_pc_final = shifted_sc.mpcc.min_per_channel, shifted_sc.mpcc.max_per_channel
        restored_min_pc = min_pc_final - shifting_factor
        restored_max_pc = max_pc_final - shifting_factor
        self.assertTrue(np.allclose(restored_min_pc, min_pc))
        self.assertTrue(np.allclose(restored_max_pc, max_pc))

    def test_minmax_shift_per_tensor_init_max(self, num_of_shifting_factors=1):
        sc = init_stats_container(num_of_shifting_factors, init_max=99)
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel
        shifted_sc, shifting_factor = shift_stats_container(sc, num_of_shifting_factors)
        min_pc_final, max_pc_final = shifted_sc.mpcc.min_per_channel, shifted_sc.mpcc.max_per_channel
        restored_min_pc = min_pc_final - shifting_factor
        restored_max_pc = max_pc_final - shifting_factor
        self.assertTrue(np.allclose(restored_min_pc, min_pc))
        self.assertTrue(np.allclose(restored_max_pc, max_pc))

    def test_minmax_shift_per_channel_init_minmax(self, num_of_shifting_factors=10):
        sc = init_stats_container(num_of_shifting_factors, init_min=0, init_max=99)
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel
        shifted_sc, shifting_factor = shift_stats_container(sc, num_of_shifting_factors)
        min_pc_final, max_pc_final = shifted_sc.mpcc.min_per_channel, shifted_sc.mpcc.max_per_channel
        restored_min_pc = min_pc_final - shifting_factor
        restored_max_pc = max_pc_final - shifting_factor
        self.assertTrue(np.allclose(restored_min_pc, min_pc))
        self.assertTrue(np.allclose(restored_max_pc, max_pc))

        min_final, max_final = shifted_sc.get_min_max_values()
        # init values should be ignored as stats were shifted per-channel
        self.assertTrue(np.allclose(min_final, np.min(min_pc + shifting_factor)))
        self.assertTrue(np.allclose(max_final, np.max(max_pc + shifting_factor)))

    def test_minmax_shift_per_tensor_init_minmax(self, num_of_shifting_factors=1):
        sc = init_stats_container(num_of_shifting_factors, init_min=0, init_max=99)
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel
        shifted_sc, shifting_factor = shift_stats_container(sc, num_of_shifting_factors)
        min_pc_final, max_pc_final = shifted_sc.mpcc.min_per_channel, shifted_sc.mpcc.max_per_channel
        restored_min_pc = min_pc_final - shifting_factor
        restored_max_pc = max_pc_final - shifting_factor
        self.assertTrue(np.allclose(restored_min_pc, min_pc))
        self.assertTrue(np.allclose(restored_max_pc, max_pc))

    ########### Test scaling for collector with init values ###########
    def test_minmax_scale_per_channel_init_min(self, num_of_scaling_factors=10):
        sc = init_stats_container(num_of_scaling_factors, init_min=0)
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel
        shifted_sc, shifting_factor = scale_stats_container(sc, num_of_scaling_factors)
        min_pc_final, max_pc_final = shifted_sc.mpcc.min_per_channel, shifted_sc.mpcc.max_per_channel
        restored_min_pc = min_pc_final / shifting_factor
        restored_max_pc = max_pc_final / shifting_factor
        self.assertTrue(np.allclose(restored_min_pc, min_pc))
        self.assertTrue(np.allclose(restored_max_pc, max_pc))

        min_final, max_final = shifted_sc.get_min_max_values()
        self.assertTrue(np.allclose(min_final, np.min(min_pc * shifting_factor)))
        self.assertTrue(np.allclose(max_final, np.max(max_pc * shifting_factor)))

    def test_minmax_scale_per_channel_init_max(self, num_of_scaling_factors=10):
        sc = init_stats_container(num_of_scaling_factors, init_max=99)
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel
        shifted_sc, shifting_factor = scale_stats_container(sc, num_of_scaling_factors)
        min_pc_final, max_pc_final = shifted_sc.mpcc.min_per_channel, shifted_sc.mpcc.max_per_channel
        restored_min_pc = min_pc_final / shifting_factor
        restored_max_pc = max_pc_final / shifting_factor
        self.assertTrue(np.allclose(restored_min_pc, min_pc))
        self.assertTrue(np.allclose(restored_max_pc, max_pc))

        min_final, max_final = shifted_sc.get_min_max_values()
        self.assertTrue(np.allclose(max_final, np.max(max_pc * shifting_factor)))
        self.assertTrue(np.allclose(min_final, np.min(min_pc * shifting_factor)))

    def test_minmax_scale_per_tensor_init_min(self, num_of_scaling_factors=1):
        sc = init_stats_container(num_of_scaling_factors, init_min=0)
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel
        shifted_sc, shifting_factor = scale_stats_container(sc, num_of_scaling_factors)
        min_pc_final, max_pc_final = shifted_sc.mpcc.min_per_channel, shifted_sc.mpcc.max_per_channel
        restored_min_pc = min_pc_final / shifting_factor
        restored_max_pc = max_pc_final / shifting_factor
        self.assertTrue(np.allclose(restored_min_pc, min_pc))
        self.assertTrue(np.allclose(restored_max_pc, max_pc))

    def test_minmax_scale_per_tensor_init_max(self, num_of_scaling_factors=1):
        sc = init_stats_container(num_of_scaling_factors, init_max=99)
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel
        shifted_sc, shifting_factor = scale_stats_container(sc, num_of_scaling_factors)
        min_pc_final, max_pc_final = shifted_sc.mpcc.min_per_channel, shifted_sc.mpcc.max_per_channel
        restored_min_pc = min_pc_final / shifting_factor
        restored_max_pc = max_pc_final / shifting_factor
        self.assertTrue(np.allclose(restored_min_pc, min_pc))
        self.assertTrue(np.allclose(restored_max_pc, max_pc))

    def test_minmax_scale_per_channel_init_minmax(self, num_of_scaling_factors=10):
        sc = init_stats_container(num_of_scaling_factors, init_min=0, init_max=99)
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel
        shifted_sc, shifting_factor = scale_stats_container(sc, num_of_scaling_factors)
        min_pc_final, max_pc_final = shifted_sc.mpcc.min_per_channel, shifted_sc.mpcc.max_per_channel
        restored_min_pc = min_pc_final / shifting_factor
        restored_max_pc = max_pc_final / shifting_factor
        self.assertTrue(np.allclose(restored_min_pc, min_pc))
        self.assertTrue(np.allclose(restored_max_pc, max_pc))

        min_final, max_final = shifted_sc.get_min_max_values()
        self.assertTrue(np.allclose(min_final, np.min(min_pc * shifting_factor)))
        self.assertTrue(np.allclose(max_final, np.max(max_pc * shifting_factor)))

    def test_minmax_scale_per_tensor_init_minmax(self, num_of_scaling_factors=1):
        sc = init_stats_container(num_of_scaling_factors, init_min=0, init_max=99)
        min_pc, max_pc = sc.mpcc.min_per_channel, sc.mpcc.max_per_channel
        shifted_sc, shifting_factor = scale_stats_container(sc, num_of_scaling_factors)
        min_pc_final, max_pc_final = shifted_sc.mpcc.min_per_channel, shifted_sc.mpcc.max_per_channel
        restored_min_pc = min_pc_final / shifting_factor
        restored_max_pc = max_pc_final / shifting_factor
        self.assertTrue(np.allclose(restored_min_pc, min_pc))
        self.assertTrue(np.allclose(restored_max_pc, max_pc))


if __name__ == '__main__':
    unittest.main()
