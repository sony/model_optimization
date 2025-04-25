#  Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

import pytest
from unittest.mock import Mock

from model_compression_toolkit.core.common.fusion.fusing_info import FusingInfoGenerator, FUSED_OP_ID_PREFIX, FusingInfo
from model_compression_toolkit.target_platform_capabilities import FrameworkQuantizationCapabilities
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.constants import FUSED_LAYER_PATTERN, FUSED_OP_QUANT_CONFIG

from tests.common_tests.helpers.generate_test_tpc import generate_test_attr_configs, generate_test_op_qc

# Setup TEST_QC and TEST_QCO for testing.
TEST_QC = generate_test_op_qc(**generate_test_attr_configs())


class MockBaseNode:
    """
    Mock implementation of a base graph node.
    Allows for equality checks and hashing based on the node name.
    """

    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, MockBaseNode) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


@pytest.fixture
def fusing_patterns():
    """
    - Returns predefined fusing patterns: Conv2D + ReLU and Linear + Softmax.
    """
    return [{FUSED_LAYER_PATTERN: ["Conv2d", "ReLU"], FUSED_OP_QUANT_CONFIG: None},
            {FUSED_LAYER_PATTERN: ["Linear", "Softmax"], FUSED_OP_QUANT_CONFIG: None}]


@pytest.fixture
def mock_nodes():
    """
    Creates mock nodes representing a simple neural network structure.
    - Nodes: Conv2D, ReLU, Linear, Softmax.
    """
    node1 = Mock(spec=BaseNode)
    node1.name = "conv"
    node1.layer_class = "Conv2d"

    node2 = Mock(spec=BaseNode)
    node2.name = "relu"
    node2.layer_class = "ReLU"

    node3 = Mock(spec=BaseNode)
    node3.name = "linear"
    node3.layer_class = "Linear"

    node4 = Mock(spec=BaseNode)
    node4.name = "softmax"
    node4.layer_class = "Softmax"

    return [node1, node2, node3, node4]


@pytest.fixture
def mock_graph(mock_nodes):
    """
    Creates a mock graph with topologically sorted nodes and defined connectivity.
    - Implements `get_next_nodes` and `get_prev_nodes` to maintain linear order.
    """
    graph = Mock()
    graph.nodes.return_value = mock_nodes
    graph.get_topo_sorted_nodes.return_value = mock_nodes

    adjacency = {
        mock_nodes[0]: [mock_nodes[1]],  # conv -> relu
        mock_nodes[1]: [mock_nodes[2]],  # relu -> linear
        mock_nodes[2]: [mock_nodes[3]],  # linear -> softmax
        mock_nodes[3]: []               # softmax has no outputs
    }

    reverse_adjacency = {
        mock_nodes[0]: [],               # conv has no inputs
        mock_nodes[1]: [mock_nodes[0]],  # relu <- conv
        mock_nodes[2]: [mock_nodes[1]],  # linear <- relu
        mock_nodes[3]: [mock_nodes[2]]   # softmax <- linear
    }

    graph.get_next_nodes.side_effect = lambda node: adjacency.get(node, [])
    graph.get_prev_nodes.side_effect = lambda node: reverse_adjacency.get(node, [])

    return graph


@pytest.fixture
def fusing_info_generator(fusing_patterns):
    """
    Creates a FusingInfoGenerator using the fusing patterns.
    """
    return FusingInfoGenerator(fusing_patterns)


def test_fusing_info_number_of_operations(mock_graph, fusing_info_generator):
    """
    Tests that the correct number of fused operations is detected.
    - Expects 2 fused operations: Conv2D + ReLU, Linear + Softmax.
    """
    fi = fusing_info_generator.generate_fusing_info(mock_graph)
    fused_operations = fi.get_all_fused_operations()
    assert len(fused_operations) == 2, "Expected 2 fused operations"


def test_fusing_info_operation_contents(mock_graph, fusing_info_generator, mock_nodes):
    """
    Tests that the fused operations contain the correct node groups.
    - Checks that the correct node names are assigned to each fused operation.
    """
    fi = fusing_info_generator.generate_fusing_info(mock_graph)
    fused_operations = fi.get_all_fused_operations()

    expected_op1_id = f"{FUSED_OP_ID_PREFIX}conv_relu"
    expected_op2_id = f"{FUSED_OP_ID_PREFIX}linear_softmax"

    assert expected_op1_id in fused_operations, f"{expected_op1_id} not found"
    assert expected_op2_id in fused_operations, f"{expected_op2_id} not found"

    conv_node, relu_node, linear_node, softmax_node = mock_nodes

    assert [f.name for f in fused_operations[expected_op1_id]] == [conv_node.name,
                                                                   relu_node.name], "Incorrect nodes in first fused operation"
    assert [f.name for f in fused_operations[expected_op2_id]] == [linear_node.name,
                                                                   softmax_node.name], "Incorrect nodes in second fused operation"


def test_fusing_info_node_mapping(mock_graph, fusing_info_generator, mock_nodes):
    """
    Tests that each node is correctly mapped to its fused operation.
    """
    fi = fusing_info_generator.generate_fusing_info(mock_graph)
    node_to_fused_map = fi.get_node_to_fused_node_map()

    conv_node, relu_node, linear_node, softmax_node = mock_nodes

    expected_op1_id = f"{FUSED_OP_ID_PREFIX}conv_relu"
    expected_op2_id = f"{FUSED_OP_ID_PREFIX}linear_softmax"

    assert node_to_fused_map[conv_node.name] == expected_op1_id
    assert node_to_fused_map[relu_node.name] == expected_op1_id
    assert node_to_fused_map[linear_node.name] == expected_op2_id
    assert node_to_fused_map[softmax_node.name] == expected_op2_id


def test_fusing_info_validation(mock_graph, fusing_info_generator):
    """
    Tests that the fusing info successfully validates a correct graph.
    - If validation raises an error, the test fails.
    """
    fi = fusing_info_generator.generate_fusing_info(mock_graph)
    fi.validate(mock_graph)


def test_fusing_info_validation_failure_topology_change(mock_graph, fusing_info_generator, mock_nodes):
    """
    Tests that validation fails when the graph topology is altered incorrectly.
    - Adds an extra node, creating multiple successors for a node.
    - Expects validation to fail with a ValueError.
    """
    fusing_info = fusing_info_generator.generate_fusing_info(mock_graph)
    extra_node = Mock(spec=BaseNode)
    extra_node.name = 'extra_node_name'

    def modified_get_next_nodes(node):
        if node == mock_nodes[0]:
            return [mock_nodes[1], extra_node]  # Conv now has two successors
        return []

    mock_graph.get_next_nodes.side_effect = modified_get_next_nodes

    with pytest.raises(ValueError):
        fusing_info.validate(mock_graph)

def test_add_fused_operation_adds_data(mock_graph, fusing_info_generator):
    fi = FusingInfo()
    node1 = MockBaseNode("a")
    node2 = MockBaseNode("b")
    op_id = f"{FUSED_OP_ID_PREFIX}a_b"
    fi.add_fused_operation(op_id, (node1, node2))

    assert op_id in fi.get_all_fused_operations()
    assert fi.get_fused_node_name("a") == op_id
    assert fi.get_fused_node_name("b") == op_id

def test_remove_fused_operation_raises_for_missing_op(mock_graph, fusing_info_generator):
    fi = FusingInfo()
    with pytest.raises(ValueError, match="Fused operation __fused__missing does not exist"):
        fi.remove_fused_operation("__fused__missing")

def test_is_node_in_fused_op_returns_true_for_present_node(mock_graph, fusing_info_generator):
    node1 = MockBaseNode("a")
    node2 = MockBaseNode("b")
    fi = FusingInfo(fusing_data={f"{FUSED_OP_ID_PREFIX}a_b": (node1, node2)})

    assert fi.is_node_in_fused_op(node1)
    assert fi.is_node_in_fused_op(node2)

def test_is_node_in_fused_op_returns_false_for_absent_node(mock_graph, fusing_info_generator):
    node1 = MockBaseNode("a")
    node2 = MockBaseNode("b")
    fi = FusingInfo(fusing_data={f"{FUSED_OP_ID_PREFIX}a_b": (node1, node2)})

    unrelated = MockBaseNode("unrelated")
    assert not fi.is_node_in_fused_op(unrelated)



def create_mock_base_node(name: str, layer_class: str):
    """
    Function for creating the mock nodes required for a simple neural network structure.
    Enables node name, layer class, type, and type checking method.
    """

    dummy_initalize = {'framework_attr': {},
                       'input_shape': (),
                       'output_shape': (),
                       'weights': {}}

    real_node = BaseNode(name=name, layer_class=layer_class, **dummy_initalize)

    node = Mock(spec=real_node)
    node.is_match_type = real_node.is_match_type
    node.layer_class = layer_class
    node.name = name

    return node

@pytest.fixture
def fusing_patterns_with_qconfig():
    """
    - Returns predefined fusing patterns: Conv2D + ReLU and  Conv2D + Tanh, Linear + Softmax.
    """
    return [{FUSED_LAYER_PATTERN: ["Conv2d", "ReLU"], FUSED_OP_QUANT_CONFIG: TEST_QC},
            {FUSED_LAYER_PATTERN: ["Conv2d", "Tanh"], FUSED_OP_QUANT_CONFIG: None}, 
            {FUSED_LAYER_PATTERN: ["Linear", "Softmax"], FUSED_OP_QUANT_CONFIG: TEST_QC }]

@pytest.fixture
def fusing_info_generator_with_qconfig(fusing_patterns_with_qconfig):
    """
    Creates a FusingInfoGenerator using the fusing patterns.
    """
    return FusingInfoGenerator(fusing_patterns_with_qconfig)

@pytest.fixture
def mock_qconfig_set_nodes():
    """
    Creates mock nodes representing a simple neural network structure.
    - Nodes: Conv2D, ReLU, Conv2D, Tanh, Linear, Softmax.
    """
    node1 = create_mock_base_node(name='conv', layer_class='Conv2d')
    node2 = create_mock_base_node(name='relu', layer_class='ReLU')
    node3 = create_mock_base_node(name='conv_2', layer_class='Conv2d')
    node4 = create_mock_base_node(name='tanh', layer_class='Tanh')
    node5 = create_mock_base_node(name='linear', layer_class='Linear')
    node6 = create_mock_base_node(name='softmax', layer_class='Softmax')

    return [node1, node2, node3, node4, node5, node6]


@pytest.fixture
def mock_qconfig_set_graph(mock_qconfig_set_nodes):
    """
    Creates a mock graph with topologically sorted nodes and defined connectivity.
    - Implements `get_next_nodes` and `get_prev_nodes` to maintain linear order.
    """
    mock_nodes = mock_qconfig_set_nodes

    graph = Mock()
    graph.nodes.return_value = mock_nodes
    graph.get_topo_sorted_nodes.return_value = mock_nodes

    adjacency = {
        mock_nodes[0]: [mock_nodes[1]],  # conv -> relu
        mock_nodes[1]: [mock_nodes[2]],  # relu -> conv_2
        mock_nodes[2]: [mock_nodes[3]],  # conv_2 -> silu
        mock_nodes[3]: [mock_nodes[4]],  # silu -> linear
        mock_nodes[4]: [mock_nodes[5]],  # linear -> softmax
        mock_nodes[5]: []                # softmax has no outputs
    }

    reverse_adjacency = {
        mock_nodes[0]: [],               # conv has no inputs
        mock_nodes[1]: [mock_nodes[0]],  # relu <- conv
        mock_nodes[2]: [mock_nodes[1]],  # conv_2 <- relu
        mock_nodes[3]: [mock_nodes[2]],  # silu <- conv_2
        mock_nodes[4]: [mock_nodes[3]],  # linear <- silu
        mock_nodes[5]: [mock_nodes[4]]   # softmax <- linear
    }

    graph.get_next_nodes.side_effect = lambda node: adjacency.get(node, [])
    graph.get_prev_nodes.side_effect = lambda node: reverse_adjacency.get(node, [])

    return graph


def test_fusing_info_qconfig_mapping(mock_qconfig_set_graph, fusing_info_generator_with_qconfig):
    """
    Tests that each node is correctly mapped to its fused quantization configs.
    """
    fi = fusing_info_generator_with_qconfig.generate_fusing_info(mock_qconfig_set_graph)
    fi_qconfig_map = fi.get_fusing_quantization_config_map()

    expected_op1_id = f"{FUSED_OP_ID_PREFIX}conv_relu"
    expected_op2_id = f"{FUSED_OP_ID_PREFIX}conv_2_tanh"
    expected_op3_id = f"{FUSED_OP_ID_PREFIX}linear_softmax"

    assert len(fi_qconfig_map) == 3
    assert fi_qconfig_map[expected_op1_id] == TEST_QC
    assert fi_qconfig_map[expected_op2_id] == None
    assert fi_qconfig_map[expected_op3_id] == TEST_QC


def test_add_fused_operation_adds_data_and_qconfig(mock_qconfig_set_graph, fusing_info_generator_with_qconfig):
    """
    Tests whether the added node is correctly assigned the fused quantization config.
    """

    fi = fusing_info_generator_with_qconfig.generate_fusing_info(mock_qconfig_set_graph)
    fi_qconfig_map = fi.get_fusing_quantization_config_map()

    ### Checking the number of mappings before addition
    assert len(fi_qconfig_map) == 3

    node1 = create_mock_base_node(name='conv_a', layer_class='Conv2d')
    node2 = create_mock_base_node(name='relu_b', layer_class='ReLU')

    op_id = f"{FUSED_OP_ID_PREFIX}conv_a_relu_b"
    fi.add_fused_operation(op_id, (node1, node2))
    fi_qconfig_map = fi.get_fusing_quantization_config_map()

    ### Checking the mapping information after addition
    assert op_id in fi.get_all_fused_operations()
    assert fi.get_fused_node_name("conv_a") == op_id
    assert fi.get_fused_node_name("relu_b") == op_id

    assert len(fi_qconfig_map) == 4
    assert fi.get_fused_op_quantization_config(op_id) == TEST_QC


def test_remove_fusing_data_and_qconfig(mock_qconfig_set_graph, fusing_info_generator_with_qconfig, mock_qconfig_set_nodes):
    """
    Tests that the fused quantization config for the specified operation is removed from the map.
    """

    fi = fusing_info_generator_with_qconfig.generate_fusing_info(mock_qconfig_set_graph)

    ### Delete Conv2D + ReLU pattern.
    conv_node, relu_node, _, _, _, _ = mock_qconfig_set_nodes
    op_id = f"{FUSED_OP_ID_PREFIX}conv_relu"

    ### Checking the mapping information before deletion.
    assert len(fi.get_fusing_quantization_config_map()) == 3
    assert fi.get_fused_op_quantization_config(op_id) == TEST_QC
    assert fi.get_fused_nodes(op_id) == (conv_node, relu_node)

    fi.remove_fused_operation(op_id)
    fi_qconfig_map = fi.get_fusing_quantization_config_map()

    ### Checking the mapping information after deletion.
    assert len(fi.get_fusing_quantization_config_map()) == 2
    assert fi.get_fused_op_quantization_config(op_id) == None
    assert fi.get_fused_nodes(op_id) == None
