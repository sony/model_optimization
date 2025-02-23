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

from model_compression_toolkit.core import ResourceUtilization
from model_compression_toolkit.core.common.mixed_precision.solution_refinement_procedure import greedy_solution_refinement_procedure

import pytest
from unittest.mock import Mock, MagicMock
import copy


@pytest.fixture
def search_manager():
    manager = Mock()
    manager.graph.get_configurable_sorted_nodes = MagicMock()
    manager.fw_info.get_kernel_op_attributes = MagicMock()
    manager.replace_config_in_index = MagicMock(
        side_effect=lambda config, idx, candidate: (
            lambda new_config: (new_config.__setitem__(idx, candidate), new_config)[1]
        )(copy.deepcopy(config))
    )
    return manager


@pytest.fixture
def candidate_configs():
    def _create_candidates(weight_bits_0, act_bits_0, weight_bits_1, act_bits_1):
        candidate_0 = Mock()
        candidate_0.weights_quantization_cfg = Mock()
        candidate_0.weights_quantization_cfg.get_attr_config = MagicMock(
            return_value=Mock(weights_n_bits=weight_bits_0)
        )
        candidate_0.activation_quantization_cfg = Mock()
        candidate_0.activation_quantization_cfg.activation_n_bits = act_bits_0

        candidate_1 = Mock()
        candidate_1.weights_quantization_cfg = Mock()
        candidate_1.weights_quantization_cfg.get_attr_config = MagicMock(
            return_value=Mock(weights_n_bits=weight_bits_1)
        )
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
    "fw_kernel, resource_limit, alternative_candidate_resources_usage", [
        (['weights'], {'weights_memory': 80}, {'weights_memory':0, 'activation_memory':0}),
        (['weights'], {'weights_memory': 80, 'activation_memory': 80}, {'weights_memory':0, 'activation_memory':100}),
        ([None], {'activation_memory': 80}, {'weights_memory':100, 'activation_memory':0}),
        ([None], {'weights_memory': 80, 'activation_memory': 80}, {'weights_memory':100, 'activation_memory':0}),
    ])
def test_greedy_solution_refinement_procedure(
        search_manager,
        candidate_configs,
        fw_kernel,
        resource_limit,
        alternative_candidate_resources_usage
):
    weight_bits_0 = 8
    act_bits_0 = 16
    weight_bits_1 = 4
    act_bits_1 = 8

    initial_solution = [1]
    expected_solution = [1]

    node_mock = Mock()
    node_mock.candidates_quantization_cfg = candidate_configs(weight_bits_0, act_bits_0, weight_bits_1, act_bits_1)

    search_manager.graph.get_configurable_sorted_nodes.return_value = [node_mock]
    search_manager.fw_info.get_kernel_op_attributes.return_value = fw_kernel

    search_manager.compute_resource_utilization_for_config = MagicMock(side_effect=lambda config: {
        0: ResourceUtilization(**alternative_candidate_resources_usage),
        1: ResourceUtilization(weights_memory=50, activation_memory=50),
    }[config[-1]])

    target_resource_utilization = ResourceUtilization(**resource_limit)

    refined_solution = greedy_solution_refinement_procedure(
        mp_solution=initial_solution,
        search_manager=search_manager,
        target_resource_utilization=target_resource_utilization
    )

    assert refined_solution == expected_solution
