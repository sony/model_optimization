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

import pytest
from mct_quantizers import QuantizationMethod

from model_compression_toolkit.core.common.quantization.node_quantization_config import \
    NodeActivationQuantizationConfig, ActivationQuantizationMode
from model_compression_toolkit.target_platform_capabilities import OpQuantizationConfig


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

    def test_quantization_mode(self):
        with pytest.raises(ValueError):
            NodeActivationQuantizationConfig(self._get_op_config(True, True))
        assert (NodeActivationQuantizationConfig(self._get_op_config(False, False)).
                quant_mode == ActivationQuantizationMode.NO_QUANT)
        assert (NodeActivationQuantizationConfig(self._get_op_config(True, False)).
                quant_mode == ActivationQuantizationMode.QUANT)
        assert (NodeActivationQuantizationConfig(self._get_op_config(False, True)).
                quant_mode == ActivationQuantizationMode.PRESERVE_QUANT)
