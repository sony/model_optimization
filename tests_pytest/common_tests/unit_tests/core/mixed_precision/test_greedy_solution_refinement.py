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
from typing import Dict, List

from model_compression_toolkit.core import ResourceUtilization
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_manager import \
    MixedPrecisionSearchManager
from model_compression_toolkit.core.common.mixed_precision.solution_refinement_procedure import greedy_solution_refinement_procedure

import pytest
from unittest.mock import Mock, MagicMock
import copy


@pytest.fixture
def search_manager():
    manager = Mock()
    manager.mp_topo_configurable_nodes = MagicMock()
    manager.fw_info.get_kernel_op_attribute = MagicMock()
    manager.copy_config_with_replacement = MixedPrecisionSearchManager.copy_config_with_replacement
    manager.using_virtual_graph = False    # asserted in refinement
    return manager


@pytest.fixture
def candidate_configs():
    def _create_candidates(weight_bits_dict_0: Dict[str, int], act_bits_0: int,
                          weight_bits_dict_1: Dict[str, int], act_bits_1: int) -> List[Mock]:
        """
        Create mock candidates with dictionaries of weight bit-widths and activation bit-widths.

        Args:
            weight_bits_dict_0: Dictionary of weight attribute names to bit-widths for candidate 0.
            act_bits_0: Activation bit-width for candidate 0.
            weight_bits_dict_1: Dictionary of weight attribute names to bit-widths for candidate 1.
            act_bits_1: Activation bit-width for candidate 1.

        Returns:
            List of two mock candidates.
        """
        # Candidate 0
        candidate_0 = Mock()
        candidate_0.weights_quantization_cfg = Mock()
        # Mock all_weight_attrs as the list of keys from the dictionary
        candidate_0.weights_quantization_cfg.all_weight_attrs = list(weight_bits_dict_0.keys())
        # Mock a method or property to return the full weight bits dictionary
        candidate_0.weights_quantization_cfg.get_attr_config = MagicMock(
            return_value=Mock(weights_n_bits=weight_bits_dict_0['kernel']))
        candidate_0.activation_quantization_cfg = Mock()
        candidate_0.activation_quantization_cfg.activation_n_bits = act_bits_0

        # Candidate 1
        candidate_1 = Mock()
        candidate_1.weights_quantization_cfg = Mock()
        candidate_1.weights_quantization_cfg.all_weight_attrs = list(weight_bits_dict_1.keys())
        candidate_1.weights_quantization_cfg.get_attr_config = MagicMock(
            return_value=Mock(weights_n_bits=weight_bits_dict_1['kernel']))
        candidate_1.activation_quantization_cfg = Mock()
        candidate_1.activation_quantization_cfg.activation_n_bits = act_bits_1

        return [candidate_0, candidate_1]

    return _create_candidates


# The four test cases covered below:
# 1. Ensuring that a valid candidate is not found when weight grows from 4bits to 8bits.
# 2. Checking if a valid candidate is found but doesn't satisfy the RU limitation when weight grows from 4bits to 8bits.
# 3. Ensuring that no valid candidate exists when activation grows from 8bits to 16bits.
# 4. Verifying that a valid candidate is found but does not satisfy the RU limitation when activation grows from 8bits to 16bits.
@pytest.mark.parametrize(
    "resource_limit, alternative_candidate_resources_usage", [
        ({'weights_memory': 80}, {'weights_memory':0, 'activation_memory':0}),
        ({'weights_memory': 80, 'activation_memory': 80}, {'weights_memory':0, 'activation_memory':100}),
        ({'activation_memory': 80}, {'weights_memory':100, 'activation_memory':0}),
        ({'weights_memory': 80, 'activation_memory': 80}, {'weights_memory':100, 'activation_memory':0}),
    ])
def test_greedy_solution_refinement_procedure(
        search_manager,
        candidate_configs,
        resource_limit,
        alternative_candidate_resources_usage
):
    weight_bits_dict_0 = {'kernel': 8}
    act_bits_0 = 16
    weight_bits_dict_1 = {'kernel': 4}
    act_bits_1 = 8

    node_mock = Mock()
    node_mock.candidates_quantization_cfg = candidate_configs(weight_bits_dict_0, act_bits_0, weight_bits_dict_1, act_bits_1)

    initial_solution = {node_mock: 1}
    expected_solution = {node_mock: 1}

    search_manager.mp_topo_configurable_nodes = [node_mock]

    search_manager.compute_resource_utilization_for_config = MagicMock(side_effect=lambda config: {
        0: ResourceUtilization(**alternative_candidate_resources_usage),
        1: ResourceUtilization(weights_memory=50, activation_memory=50),
    }[config[node_mock]])

    target_resource_utilization = ResourceUtilization(**resource_limit)

    refined_solution = greedy_solution_refinement_procedure(
        mp_solution=initial_solution,
        search_manager=search_manager,
        target_resource_utilization=target_resource_utilization
    )

    assert refined_solution == expected_solution
