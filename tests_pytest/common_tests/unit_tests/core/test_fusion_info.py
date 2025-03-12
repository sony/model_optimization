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

from model_compression_toolkit.core.common.fusion.fusing_info import FusingInfoGenerator
from model_compression_toolkit.target_platform_capabilities import FrameworkQuantizationCapabilities
from model_compression_toolkit.core.common import BaseNode


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
def mock_fqc():
    """
    Creates a mock FrameworkQuantizationCapabilities object.
    - Returns predefined fusing patterns: Conv2D + ReLU and Linear + Softmax.
    """
    mock = Mock(spec=FrameworkQuantizationCapabilities)
    mock.get_fusing_patterns.return_value = [["Conv2d", "ReLU"],
                                             ["Linear", "Softmax"]]
    return mock


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

    def get_next_nodes(node):
        if node in mock_nodes:
            index = mock_nodes.index(node)
            return [mock_nodes[index + 1]] if index < len(mock_nodes) - 1 else []
        return []

    def get_prev_nodes(node):
        if node in mock_nodes:
            index = mock_nodes.index(node)
            return [mock_nodes[index - 1]] if index > 0 else []
        return []

    graph.get_next_nodes.side_effect = get_next_nodes
    graph.get_prev_nodes.side_effect = get_prev_nodes

    return graph


@pytest.fixture
def fusing_info_generator(mock_fqc):
    """
    Creates a FusingInfoGenerator using the mock quantization capabilities.
    """
    return FusingInfoGenerator(mock_fqc)


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

    expected_op1_id = "FusedNode_conv_relu"
    expected_op2_id = "FusedNode_linear_softmax"

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

    expected_op1_id = "FusedNode_conv_relu"
    expected_op2_id = "FusedNode_linear_softmax"

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


def test_fusing_info_disable_act_quantization(mock_graph, fusing_info_generator, mock_nodes):
    """
    Tests that the correct nodes have activation quantization disabled.
    - Expects only Conv2D and Linear nodes to have activation quantization disabled.
    """
    fi = fusing_info_generator.generate_fusing_info(mock_graph)
    nodes_to_disable = fi.get_nodes_to_disable_act_quantization()
    conv_node, _, linear_node, _ = mock_nodes
    assert nodes_to_disable == [conv_node, linear_node], "Incorrect nodes to disable activation quantization"


def test_fusing_info_validation_failure_topology_change(mock_graph, fusing_info_generator, mock_nodes):
    """
    Tests that validation fails when the graph topology is altered incorrectly.
    - Adds an extra node, creating multiple successors for a node.
    - Expects validation to fail with a ValueError.
    """
    fusing_info = fusing_info_generator.generate_fusing_info(mock_graph)
    extra_node = Mock(spec=BaseNode)

    def modified_get_next_nodes(node):
        if node == mock_nodes[0]:
            return [mock_nodes[1], extra_node]  # Conv now has two successors
        return []

    mock_graph.get_next_nodes.side_effect = modified_get_next_nodes

    with pytest.raises(ValueError, match="does not form a linear chain"):
        fusing_info.validate(mock_graph)
