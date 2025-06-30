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
from unittest.mock import MagicMock, Mock

from model_compression_toolkit.core import ResourceUtilization
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_candidates_filter import \
    filter_candidates_for_mixed_precision
from model_compression_toolkit.target_platform_capabilities import QuantizationConfigOptions
from tests_pytest._test_util.graph_builder_utils import build_node, build_nbits_qc


@pytest.mark.parametrize(
    "tru, expected_act_candidates, expected_weight_candidates", [
        (ResourceUtilization(weights_memory=1), 1, 2),  # Only weights restricted
        (ResourceUtilization(activation_memory=1), 2, 1),  # Only activations restricted
        (ResourceUtilization(weights_memory=1, activation_memory=1), 2, 2),  # Both restricted
        (ResourceUtilization(), 2, 2),  # Neither restricted
        (ResourceUtilization(total_memory=1), 2, 2),  # Total memory restricted
        (ResourceUtilization(bops=1), 2, 2),  # BOPS restricted
    ])
def test_filtering(graph_mock, tru, expected_act_candidates, expected_weight_candidates, patch_fw_info):
    """
    Test candidate filtering based on weight and activation restrictions.

    This test verifies that:
    - When weight restriction is applied, only 1 activation candidate remains.
    - When activation restriction is applied, only 1 weight candidate remains.
    - When both restrictions are applied, filtering occurs on both activations and weights.
    - When no restrictions are applied, all candidates remain.
    """
    patch_fw_info.get_kernel_op_attribute = Mock(return_value='kernel')

    act_node = build_node(qcs=[build_nbits_qc(8), build_nbits_qc(4)])
    act_node.get_qco = Mock(return_value=Mock(spec=QuantizationConfigOptions, base_config=Mock(activation_n_bits=8)))

    # Weights-configurable node with candidates
    weight_node = build_node(qcs=[build_nbits_qc(w_attr={'kernel': (8, True)}),
                                  build_nbits_qc(w_attr={'kernel': (4, True)})])
    qco = Mock(spec=QuantizationConfigOptions,
               base_config=MagicMock(attr_weights_configs_mapping={'kernel': MagicMock(weights_n_bits=8)}))
    weight_node.get_qco = Mock(return_value=qco)

    graph_mock.get_activation_configurable_nodes.return_value = [act_node]
    graph_mock.get_weights_configurable_nodes.return_value = [weight_node]

    filter_candidates_for_mixed_precision(graph_mock, tru)

    assert len(act_node.candidates_quantization_cfg) == expected_act_candidates
    assert len(weight_node.candidates_quantization_cfg) == expected_weight_candidates
