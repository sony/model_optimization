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

from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core import QuantizationErrorMethod, QuantizationConfig
from model_compression_toolkit.core.common.collectors.statistics_collector import StatsCollector
from model_compression_toolkit.core.common.node_prior_info import NodePriorInfo
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeActivationQuantizationConfig
from model_compression_toolkit.core.common.quantization.quantization_params_generation import (
    power_of_two_no_clipping_selection_min_max,
    symmetric_no_clipping_selection_min_max,
    uniform_no_clipping_selection_min_max
)
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_activations_computation import (
    get_histogram_data,
    determine_signedness,
    update_activation_quantization_params_fn,
    get_activations_qparams
)
from model_compression_toolkit.target_platform_capabilities import Signedness, OpQuantizationConfig
from model_compression_toolkit.target_platform_capabilities.schema.v1 import AttributeQuantizationConfig


class TestActivationQParams:
    def _create_stats_container(self, out_channel_axis=1):
        """
        Helper to create a StatsCollector with mocked histogram collectors.
        """
        stats = StatsCollector(out_channel_axis=out_channel_axis)
        stats.hc = Mock(wraps=stats.hc)
        stats.weighted_hc = Mock(wraps=stats.weighted_hc)
        return stats

    def _create_activation_quant_cfg(self, quant_method, n_bits=8, signedness=Signedness.AUTO):
        """
        Helper to create a NodeActivationQuantizationConfig with default settings.
        """
        op_cfg = OpQuantizationConfig(
            default_weight_attr_config=AttributeQuantizationConfig(),
            attr_weights_configs_mapping={},
            activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
            activation_n_bits=n_bits,
            supported_input_activation_n_bits=n_bits,
            enable_activation_quantization=True,
            quantization_preserving=True,
            fixed_scale=None,
            fixed_zero_point=None,
            simd_size=None,
            signedness=signedness
        )
        qc = QuantizationConfig()
        activation_quant_cfg = NodeActivationQuantizationConfig(qc, op_cfg, None, None)
        activation_quant_cfg.activation_quantization_method = quant_method
        return activation_quant_cfg

    def test_get_histogram_data_error_method(self):
        # Test get_histogram_data:
        # Verifies that the correct histogram collector is used based on the activation error method.
        # Test cases:
        # - When activation_error_method is MSE, the normal histogram collector (hc) is called.
        # - When activation_error_method is HMSE, the weighted histogram collector (weighted_hc) is called.
        activation_quant_cfg = Mock(spec=NodeActivationQuantizationConfig)
        stats = self._create_stats_container(out_channel_axis=1)

        activation_quant_cfg.z_threshold = 1

        # Set return values for histogram collectors.
        stats.hc.get_histogram.return_value = ([1, 2, 3, 4], [10, 20, 30])
        stats.weighted_hc.get_histogram.return_value = ([5, 6, 7, 8], [40, 50, 60])

        # Test for MSE error method.
        activation_quant_cfg.activation_error_method = QuantizationErrorMethod.MSE
        get_histogram_data(activation_quant_cfg, stats)
        stats.hc.get_histogram.assert_called_once()
        stats.weighted_hc.get_histogram.assert_not_called()

        # Reset call history for the next test.
        stats.hc.get_histogram.reset_mock()
        stats.weighted_hc.get_histogram.reset_mock()

        # Test for HMSE error method.
        activation_quant_cfg.activation_error_method = QuantizationErrorMethod.HMSE
        get_histogram_data(activation_quant_cfg, stats)
        stats.weighted_hc.get_histogram.assert_called_once()
        stats.hc.get_histogram.assert_not_called()

    def test_determine_signedness(self):
        # Test determine_signedness:
        # Verifies that the function returns the correct boolean value based on the provided signedness,
        # output bounds, and minimum value.
        # Test cases:
        # - For SIGNED: should return True.
        # - For UNSIGNED: should return False.
        # - For AUTO with bounded output and varying min values: returns based on the sign of min_value.
        # - For AUTO with unbounded output: should return False.
        activation_quant_cfg = Mock(spec=NodeActivationQuantizationConfig)
        nodes_prior_info = Mock(spec=NodePriorInfo)
        bins_values = np.array([1, 2, 3, 4])
        bins_counts = np.array([10, 20, 30])

        # Test for SIGNED.
        activation_quant_cfg.signedness = Signedness.SIGNED
        assert determine_signedness(activation_quant_cfg, nodes_prior_info, 0, bins_values, bins_counts)

        # Test for UNSIGNED.
        activation_quant_cfg.signedness = Signedness.UNSIGNED
        assert not determine_signedness(activation_quant_cfg, nodes_prior_info, 0, bins_values, bins_counts)

        # Test for AUTO with bounded output.
        activation_quant_cfg.signedness = Signedness.AUTO
        nodes_prior_info.is_output_bounded = Mock(return_value=True)
        assert not determine_signedness(activation_quant_cfg, nodes_prior_info, 0, bins_values, bins_counts)
        assert not determine_signedness(activation_quant_cfg, nodes_prior_info, 1, bins_values, bins_counts)
        assert determine_signedness(activation_quant_cfg, nodes_prior_info, -1, bins_values, bins_counts)

        # Test for AUTO with unbounded output.
        nodes_prior_info.is_output_bounded.return_value = False
        assert not determine_signedness(activation_quant_cfg, nodes_prior_info, -1, bins_values, bins_counts)

        # Test for AUTO with negative bin values.
        bins_values = np.array([-1, 2, 3, 4])
        assert determine_signedness(activation_quant_cfg, nodes_prior_info, -1, bins_values, bins_counts)

    @pytest.mark.parametrize('quant_method, activation_quantization_params_fn', [
        (QuantizationMethod.POWER_OF_TWO, power_of_two_no_clipping_selection_min_max),
        (QuantizationMethod.SYMMETRIC, symmetric_no_clipping_selection_min_max),
        (QuantizationMethod.UNIFORM, uniform_no_clipping_selection_min_max)
    ])
    def test_update_activation_quantization_params_fn(self, quant_method, activation_quantization_params_fn):
        # Test update_activation_quantization_params_fn:
        # Verifies that the activation_quantization_params_fn attribute is correctly set based on the quantization method.
        # Test cases: POWER_OF_TWO, SYMMETRIC, and UNIFORM methods.
        nodes_prior_info = Mock(spec=NodePriorInfo)
        activation_quant_cfg = self._create_activation_quant_cfg(quant_method, n_bits=8)
        update_activation_quantization_params_fn(activation_quant_cfg, nodes_prior_info)
        assert activation_quant_cfg.activation_quantization_params_fn == activation_quantization_params_fn

    @pytest.mark.parametrize('quant_method, activation_quantization_params_fn, expected_result', [
        (QuantizationMethod.POWER_OF_TWO, power_of_two_no_clipping_selection_min_max,
         {'threshold': 16.0, 'is_signed': True}),
        (QuantizationMethod.SYMMETRIC, symmetric_no_clipping_selection_min_max,
         {'threshold': 9, 'is_signed': True}),
        (QuantizationMethod.UNIFORM, uniform_no_clipping_selection_min_max,
         {'range_min': -1, 'range_max': 9, 'is_signed': True})
    ])
    def test_get_activations_qparams(self, quant_method, activation_quantization_params_fn, expected_result):
        # Test get_activations_qparams:
        # Verifies that the computed activation quantization parameters match the expected result.
        # Test cases:
        # - POWER_OF_TWO: Expected parameters {'threshold': 16.0, 'is_signed': True}.
        # - SYMMETRIC: Expected parameters {'threshold': 9, 'is_signed': True}.
        # - UNIFORM: Expected parameters {'range_min': -1, 'range_max': 9, 'is_signed': True}.
        stats = self._create_stats_container(out_channel_axis=1)
        stats.get_min_max_values = Mock(return_value=(-1, 9))
        stats.hc.get_histogram.return_value = (np.array([-2, 1, 7, 9]), np.array([600, 50, 20]))
        stats.weighted_hc.get_histogram.return_value = (np.array([-2, 1, 7, 9]), np.array([600, 50, 20]))

        nodes_prior_info = Mock(spec=NodePriorInfo)
        nodes_prior_info.is_output_bounded = Mock(return_value=True)

        activation_quant_cfg = self._create_activation_quant_cfg(quant_method, n_bits=2)
        result = get_activations_qparams(activation_quant_cfg, nodes_prior_info, stats)
        assert result == expected_result
