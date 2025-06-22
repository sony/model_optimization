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
from typing import Callable, Any

import copy

import abc

import pytest


import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.core import QuantizationConfig, FrameworkInfo
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.quantization.node_quantization_config import ActivationQuantizationMode
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.graph.edge import EDGE_SOURCE_INDEX, EDGE_SINK_INDEX
from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner
from model_compression_toolkit.core.common.fusion.graph_fuser import GraphFuser
from tests_pytest._test_util.tpc_util import minimal_cfg_options


class BaseGraphWithFusingMetadataTest(abc.ABC):

    fw_impl: FrameworkImplementation
    fw_info: FrameworkInfo
    attach_to_fw_func: Callable
    layer_class_relu: Any  # needed for test_fail_validate_after_adding_node_that_adds_a_fusion

    def _data_gen(self):
        raise NotImplementedError()

    def _get_model(self):
        raise NotImplementedError()

    @pytest.fixture
    def minimal_tpc_with_fusing(self):
        """
        Fixture that provides a minimal Target Platform Capabilities (TPC) config used by the
        `graph_with_fusion_metadata` fixture.

        minimal_tpc_with_fusing is used as a fixture to provide graph_with_fusion_metadata, which is a required
        fixture for the actual test functions. While minimal_tpc_with_fusing itself isn’t used directly in tests,
        defining it as a fixture makes its usage cleaner.

        """
        return schema.TargetPlatformCapabilities(
            default_qco=minimal_cfg_options(),
            tpc_platform_type='test',
            operator_set=[schema.OperatorsSet(name=schema.OperatorSetNames.CONV),
                          schema.OperatorsSet(name=schema.OperatorSetNames.RELU),
                          schema.OperatorsSet(name=schema.OperatorSetNames.FULLY_CONNECTED),
                          schema.OperatorsSet(name=schema.OperatorSetNames.SOFTMAX)],
            fusing_patterns=[schema.Fusing(operator_groups=(schema.OperatorsSet(name=schema.OperatorSetNames.CONV),
                                                            schema.OperatorsSet(name=schema.OperatorSetNames.RELU))),
                             schema.Fusing(
                                 operator_groups=(schema.OperatorsSet(name=schema.OperatorSetNames.FULLY_CONNECTED),
                                                  schema.OperatorsSet(name=schema.OperatorSetNames.SOFTMAX))),
                             schema.Fusing(
                                 operator_groups=(schema.OperatorsSet(name=schema.OperatorSetNames.FULLY_CONNECTED),
                                                  schema.OperatorsSet(name=schema.OperatorSetNames.SOFTMAX),
                                                  schema.OperatorsSet(name=schema.OperatorSetNames.RELU)))
                             ]
        )

    @pytest.fixture
    def graph_with_fusion_metadata(self, minimal_tpc_with_fusing):
        """
        Creates a graph with fusing metadata based on a generated model and a predefined configuration.
        Ensures all required components (framework implementation, framework info, etc.) are present.
        """
        assert self._data_gen is not None
        assert self.fw_impl is not None
        assert self.attach_to_fw_func is not None

        self.fqc = self.attach_to_fw_func(minimal_tpc_with_fusing)

        graph_with_fusion_metadata = graph_preparation_runner(self._get_model(),
                                                              self._data_gen,
                                                              QuantizationConfig(),
                                                              fw_impl=self.fw_impl,
                                                              fqc=self.fqc,
                                                              mixed_precision_enable=False,
                                                              running_gptq=False)
        return graph_with_fusion_metadata

    def test_expected_fusing_info(self, graph_with_fusion_metadata):
        """
        Test that the graph contains expected metadata regard the fusing that should
        be found in the model.
        """
        actual_fi = graph_with_fusion_metadata.fusing_info
        assert len(actual_fi.get_all_fused_operations()) == 2
        assert sorted(actual_fi.get_all_fused_operations().keys()) == ['FusedNode_conv_relu', 'FusedNode_linear_softmax']
        assert actual_fi.node_name_to_fused_op_id == {'conv': 'FusedNode_conv_relu',
                                                      'relu': 'FusedNode_conv_relu',
                                                      'linear': 'FusedNode_linear_softmax',
                                                      'softmax': 'FusedNode_linear_softmax'}

    def test_disable_act_quantization(self, graph_with_fusion_metadata: Graph):
        """Tests that the correct nodes have activation quantization disabled after
        calling _disable_nodes_activation_quantization.
        """
        for node in graph_with_fusion_metadata.nodes:
            for qc in node.candidates_quantization_cfg:
                qc.activation_quantization_cfg.quant_mode = ActivationQuantizationMode.QUANT

        graph_with_fusion_metadata.override_fused_node_activation_quantization_candidates()
        disabled_nodes = [
            node.name for node in graph_with_fusion_metadata.nodes
            if all(not qc.activation_quantization_cfg.enable_activation_quantization
                   for qc in node.candidates_quantization_cfg)
        ]

        expected = ['conv', 'linear']
        assert sorted(disabled_nodes) == expected, f"Expected {expected}, but got {sorted(disabled_nodes)}"

    def test_fail_validate_after_node_removal(self, graph_with_fusion_metadata):
        """
        Tests validation failure after removing a node that is part of a fusion pattern.
        - Replaces a ReLU node with a new Tanh node.
        - Expects validation to fail because ReLU was part of a defined fusion pattern.
        """
        relu_node = graph_with_fusion_metadata.find_node_by_name('relu')[0]
        new_node = BaseNode(
            name='tanh',
            framework_attr={},
            input_shape=relu_node.input_shape,
            output_shape=relu_node.output_shape,
            weights={},
            layer_class="Tanh"
        )
        with pytest.raises(ValueError):
            graph_with_fusion_metadata.replace_node(relu_node, new_node)

    def test_fail_validate_after_topology_change(self, graph_with_fusion_metadata):
        """
        Tests validation failure after modifying the graph topology by adding an unintended edge.
        - Adds an edge from Conv2D to Flatten, creating multiple successors.
        - Expects validation to fail as the topology no longer follows expected fusing rules.
        """
        conv_node = graph_with_fusion_metadata.find_node_by_name('conv')[0]
        flatten_node = graph_with_fusion_metadata.find_node_by_name('flatten')[0]
        with pytest.raises(ValueError):
            graph_with_fusion_metadata.add_edge(conv_node, flatten_node, **{EDGE_SOURCE_INDEX: 1, EDGE_SINK_INDEX: 1})

    def test_fail_validate_after_adding_node_between_conv_to_relu(self, graph_with_fusion_metadata):
        """
        Tests validation failure after inserting a node between fused Conv2D and ReLU layers.
        - Removes the edge between Conv2D and ReLU.
        - Expects validation to fail as the fusion sequence is broken.
        """
        conv_node = graph_with_fusion_metadata.find_node_by_name('conv')[0]
        relu_node = graph_with_fusion_metadata.find_node_by_name('relu')[0]
        with pytest.raises(ValueError):
            graph_with_fusion_metadata.remove_edge(conv_node, relu_node)
        with pytest.raises(ValueError):
            graph_with_fusion_metadata.validate()

        # After updating the fusing info, make sure validation passes
        graph_with_fusion_metadata.fusing_info.remove_fused_operation('FusedNode_conv_relu')
        graph_with_fusion_metadata.validate()

    def test_valid_change_in_graph(self, graph_with_fusion_metadata):
        """
        Tests validation passes after changing the graph with a change that should not
        affect the fusing metadata.
        """
        graph_with_fusion_metadata.validate()
        # Add softmax node after current softmax
        softmax_node = graph_with_fusion_metadata.find_node_by_name('softmax')[0]
        new_softmax_node = BaseNode(
            name='new_softmax',
            framework_attr={},
            input_shape=softmax_node.output_shape,
            output_shape=softmax_node.output_shape,
            weights={},
            layer_class='softmax'
        )
        graph_with_fusion_metadata.add_node(new_softmax_node)
        graph_with_fusion_metadata.add_edge(softmax_node, new_softmax_node, **{EDGE_SOURCE_INDEX: 0, EDGE_SINK_INDEX: 0})
        graph_with_fusion_metadata.validate()

    def test_fail_validate_after_serialization_deserialization(self, graph_with_fusion_metadata):
        """
        Tests validation failure after serializing and deserializing the graph.
        - Serializes and deserializes the graph to ensure stability.
        - Breaks fusing by removing an edge and checks if validation fails.
        """
        graph_copy = copy.deepcopy(graph_with_fusion_metadata)
        graph_copy.validate()
        conv_node = graph_copy.find_node_by_name('conv')[0]
        relu_node = graph_copy.find_node_by_name('relu')[0]
        with pytest.raises(ValueError):
            graph_copy.remove_edge(conv_node, relu_node)

    def test_fail_validate_after_adding_node_that_adds_a_fusion(self, graph_with_fusion_metadata):
        """
        Tests validation failure after introducing a new fusion pattern by adding a node.
        - Adds a ReLU node after Softmax.
        - The resulting pattern FullyConnected -> Softmax -> ReLU is a defined fusion.
        - Since this pattern didn't exist in the original graph, re-running fusing should raise an error.
        """
        # Step 1: Validate original graph (should pass)
        graph_with_fusion_metadata.validate()
        fuser = GraphFuser()
        fuser.apply_node_fusion(graph_with_fusion_metadata)

        # Step 2: Add ReLU node after softmax
        softmax_node = graph_with_fusion_metadata.find_node_by_name('softmax')[0]
        relu_node = BaseNode(
            name='new_relu',
            framework_attr={},
            input_shape=softmax_node.output_shape,
            output_shape=softmax_node.output_shape,
            weights={},
            layer_class=self.layer_class_relu
        )
        graph_with_fusion_metadata.add_node(relu_node)
        graph_with_fusion_metadata.add_edge(softmax_node, relu_node, **{EDGE_SOURCE_INDEX: 0, EDGE_SINK_INDEX: 0})

        # Step 3: Run fuser and expect failure due to unexpected new fusion
        with pytest.raises(ValueError):
            fuser.apply_node_fusion(graph_with_fusion_metadata)



