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
from model_compression_toolkit.core.common.quantization.node_quantization_config import \
    NodeActivationQuantizationConfig, ActivationQuantizationMode
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
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import AttributeQuantizationConfig


class TestActivationQParams:

    def _get_op_config(self, qe, qp):
        return Mock(spec=OpQuantizationConfig,
                    default_weight_attr_config=None,
                    attr_weights_configs_mapping=None,
                    activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
                    activation_n_bits=8,
                    supported_input_activation_n_bits=[8],
                    enable_activation_quantization=qe,
                    quantization_preserving=qp,
                    fixed_scale=None,
                    fixed_zero_point=None,
                    simd_size=32,
                    signedness=None)

    def test_quantization_mode(self, quant_config_mock):
        with pytest.raises(ValueError):
            NodeActivationQuantizationConfig(quant_config_mock, self._get_op_config(True, True),
                                             lambda x: 0, lambda x: 0)
        with pytest.raises(AssertionError):
            NodeActivationQuantizationConfig(quant_config_mock, self._get_op_config(False, False),
                                             lambda x: 0, lambda x: 0).enable_activation_quantization = 6
        with pytest.raises(AssertionError):
            NodeActivationQuantizationConfig(quant_config_mock, self._get_op_config(False, False),
                                             lambda x: 0, lambda x: 0).quantization_preserving = 6
        with pytest.raises(AssertionError):
            NodeActivationQuantizationConfig(quant_config_mock, self._get_op_config(False, False),
                                             lambda x: 0, lambda x: 0).fln_quantization = 6
        naqc = NodeActivationQuantizationConfig(quant_config_mock, self._get_op_config(False, False),
                                                lambda x: 0, lambda x: 0)
        assert naqc.quant_mode == ActivationQuantizationMode.NO_QUANT
        naqc.enable_activation_quantization = True
        assert naqc.quant_mode == ActivationQuantizationMode.QUANT
        naqc.quantization_preserving = True
        assert naqc.quant_mode == ActivationQuantizationMode.PRESERVE_QUANT
        naqc.fln_quantization = True
        assert naqc.quant_mode == ActivationQuantizationMode.FLN_QUANT
