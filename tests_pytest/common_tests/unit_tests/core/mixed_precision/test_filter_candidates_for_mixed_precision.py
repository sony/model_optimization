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
from unittest.mock import MagicMock

from model_compression_toolkit.core.common.framework_info import set_fw_info
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_candidates_filter import \
    filter_candidates_for_mixed_precision


@pytest.fixture
def setup_mocks(fw_info_mock):
    """
    Set up mock objects for testing the filtering of mixed precision candidates.

    Mocks the behavior of a graph containing activation and weight-configurable nodes
    with multiple quantization candidates. The test will check whether the correct candidates
    are filtered based on given constraints.
    """
    tru = MagicMock()
    graph = MagicMock()
    set_fw_info(fw_info_mock)
    fw_info = MagicMock()
    fqc = MagicMock()

    # Activation-configurable node with candidates
    act_node = MagicMock()
    act_candidate1 = MagicMock()
    act_candidate1.activation_quantization_cfg = MagicMock(enable_activation_quantization=True, activation_n_bits=8)
    act_candidate2 = MagicMock()
    act_candidate2.activation_quantization_cfg = MagicMock(enable_activation_quantization=True, activation_n_bits=4)
    act_node.candidates_quantization_cfg = [act_candidate1, act_candidate2]
    act_node.get_qco.return_value = MagicMock(base_config=MagicMock(activation_n_bits=8))

    # Weights-configurable node with candidates
    weight_node = MagicMock()
    weight_candidate1 = MagicMock()
    weight_candidate1.weights_quantization_cfg = MagicMock()
    weight_candidate1.weights_quantization_cfg.get_attr_config = lambda attr: MagicMock(
        enable_weights_quantization=True, weights_n_bits=8) if attr == 'kernel' else None
    weight_candidate2 = MagicMock()
    weight_candidate2.weights_quantization_cfg = MagicMock()
    weight_candidate2.weights_quantization_cfg.get_attr_config = lambda attr: MagicMock(
        enable_weights_quantization=True, weights_n_bits=4) if attr == 'kernel' else None
    weight_node.candidates_quantization_cfg = [weight_candidate1, weight_candidate2]
    weight_node.kernel_attr = 'kernel'
    weight_node.get_qco.return_value = MagicMock(
        base_config=MagicMock(attr_weights_configs_mapping={'kernel': MagicMock(weights_n_bits=8)}))

    graph.get_activation_configurable_nodes.return_value = [act_node]
    graph.get_weights_configurable_nodes.return_value = [weight_node]
    fw_info.get_kernel_op_attribute = MagicMock()
    fw_info.get_kernel_op_attribute.return_value = 'kernel'

    return tru, graph, fqc, act_node, weight_node


@pytest.mark.parametrize(
    "weight_restricted, activation_restricted, expected_act_candidates, expected_weight_candidates", [
        (True, False, 1, 2),  # Only weights restricted
        (False, True, 2, 1),  # Only activations restricted
        (True, True, 2, 2),  # Both restricted
        (False, False, 2, 2),  # Neither restricted
    ])
def test_filtering(setup_mocks, weight_restricted, activation_restricted, expected_act_candidates,
                   expected_weight_candidates):
    """
    Test candidate filtering based on weight and activation restrictions.

    This test verifies that:
    - When weight restriction is applied, only 1 activation candidate remains.
    - When activation restriction is applied, only 1 weight candidate remains.
    - When both restrictions are applied, filtering occurs on both activations and weights.
    - When no restrictions are applied, all candidates remain.
    """
    tru, graph, fqc, act_node, weight_node = setup_mocks

    tru.total_mem_restricted.return_value = False
    tru.bops_restricted.return_value = False
    tru.weight_restricted.return_value = weight_restricted
    tru.activation_restricted.return_value = activation_restricted

    filter_candidates_for_mixed_precision(graph, tru, fqc)

    assert len(act_node.candidates_quantization_cfg) == expected_act_candidates
    assert len(weight_node.candidates_quantization_cfg) == expected_weight_candidates


@pytest.mark.parametrize("total_mem_restricted, bops_restricted", [
    (True, False),  # Total memory restricted
    (False, True),  # BOPS restricted
])
def test_early_return(setup_mocks, total_mem_restricted, bops_restricted):
    """
    Test early return condition when total memory or BOPS restrictions are applied.

    This test verifies that if total memory or BOPS restrictions are applied, the filtering function
    returns early without filtering candidates.

    Expected behavior:
    - No filtering occurs when total memory or BOPS constraints are enabled.
    - The number of candidates remains unchanged (2 for both activations and weights).
    """
    tru, graph, fqc, act_node, weight_node = setup_mocks

    tru.total_mem_restricted.return_value = total_mem_restricted
    tru.bops_restricted.return_value = bops_restricted
    tru.weight_restricted.return_value = True
    tru.activation_restricted.return_value = True

    filter_candidates_for_mixed_precision(graph, tru, fqc)

    assert len(act_node.candidates_quantization_cfg) == 2  # No filtering
    assert len(weight_node.candidates_quantization_cfg) == 2  # No filtering


