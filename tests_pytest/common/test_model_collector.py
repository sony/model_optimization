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
from unittest.mock import Mock, call
import numpy as np
from numpy.testing import assert_array_equal

from model_compression_toolkit.core import QuantizationErrorMethod
from model_compression_toolkit.core.common import StatsCollector, NoStatsCollector, DEFAULTCONFIG, Graph, model_collector
from model_compression_toolkit.core.common.graph.base_graph import OutTensor
from model_compression_toolkit.core.common.graph.edge import Edge
from model_compression_toolkit.core.common.hessian import HessianInfoService
from model_compression_toolkit.core.common.model_collector import create_stats_collector_for_node, create_tensor2node, ModelCollector
from tests_pytest.common.graph_builder_utils import build_node, DummyLayer, build_qc


@pytest.fixture
def fw_impl_mock():
    """
    Fixture to create a fake framework implementation with a mocked model_builder.
    """
    fw_impl = Mock()
    fw_impl.model_builder.return_value = (Mock(), None)
    return fw_impl


@pytest.fixture
def fw_info_mock():
    """
    Fixture to create a fake framework info object with a predefined out_channel_axis_mapping.
    """
    fw_info = Mock()
    fw_info.out_channel_axis_mapping = {DummyLayer: 1}
    return fw_info


class TestStatisticsCollectors:
    def test_create_stats_collector_for_node_activation_enabled(self, fw_info_mock):
        """
        Verify that for a node with activation quantization enabled,
        create_stats_collector_for_node returns a StatsCollector instance.
        """
        # Set up a fake node with activation quantization enabled and prior_info attributes.
        node = Mock()
        node.is_activation_quantization_enabled.return_value = True
        node.type = DummyLayer
        node.prior_info = Mock(min_output=-1, max_output=9)

        collector = create_stats_collector_for_node(node, fw_info_mock)
        assert isinstance(collector, StatsCollector)

    def test_create_stats_collector_for_node_activation_disabled(self, fw_info_mock):
        """
        Verify that for a node with activation quantization disabled,
        create_stats_collector_for_node returns a NoStatsCollector instance.
        """
        node = Mock()
        node.is_activation_quantization_enabled.return_value = False
        node.type = DummyLayer
        # Even if prior_info exists, it should not be used.
        node.prior_info = Mock(min_output=None, max_output=None)

        collector = create_stats_collector_for_node(node, fw_info_mock)
        assert isinstance(collector, NoStatsCollector)

    def test_create_tensor2node_assigns_stats_collector(self, fw_info_mock):
        """
        Verify that create_tensor2node assigns a new StatsCollector to a node when no valid collector exists.
        """
        # Set up a fake graph and node.
        graph = Mock()
        node = Mock()
        node.type = DummyLayer

        # Simulate absence of an output stats collector.
        graph.get_out_stats_collector = Mock(return_value=None)

        create_tensor2node(graph, node, fw_info_mock)

        # Verify that set_out_stats_collector_to_node was called with the node and a StatsCollector.
        graph.set_out_stats_collector_to_node.assert_called_once()
        args, _ = graph.set_out_stats_collector_to_node.call_args
        assigned_node, assigned_collector = args
        assert assigned_node is node
        assert isinstance(assigned_collector, StatsCollector)


class TestModelCollectorInit:
    def test_assigns_stats_collectors_to_nodes(self, fw_impl_mock, fw_info_mock):
        """
        Verify that ModelCollector.__init__ assigns appropriate statistics collectors to nodes in the graph.
        """
        # Create nodes with different activation quantization settings.
        node1 = build_node('node1', output_shape=(None, 3, 14))
        node1.is_activation_quantization_enabled = Mock(return_value=True)
        node2 = build_node('node2', output_shape=(None, 2, 71))
        node2.is_activation_quantization_enabled = Mock(return_value=False)
        node3 = build_node('node3', output_shape=(None, 59))
        node3.is_activation_quantization_enabled = Mock(return_value=True)

        # Build a graph connecting the nodes.
        graph = Graph('g',
                      input_nodes=[node1],
                      nodes=[node1, node2, node3],
                      output_nodes=[OutTensor(node3, 0)],
                      edge_list=[Edge(node1, node2, 0, 0), Edge(node2, node3, 0, 0)])
        graph.set_out_stats_collector_to_node = Mock(wraps=graph.set_out_stats_collector_to_node)

        # Simulate kernel attribute retrieval.
        fw_info_mock.get_kernel_op_attributes.return_value = [None]

        # Instantiate ModelCollector.
        mc = ModelCollector(graph, fw_impl_mock, fw_info_mock, qc=DEFAULTCONFIG)

        # Verify that stats collectors are correctly assigned.
        graph.set_out_stats_collector_to_node.assert_called()
        fw_impl_mock.model_builder.assert_called_once()
        assert isinstance(graph.get_out_stats_collector(node1), StatsCollector)
        assert isinstance(graph.get_out_stats_collector(node2), NoStatsCollector)
        assert isinstance(graph.get_out_stats_collector(node3), StatsCollector)
        assert isinstance(graph.get_in_stats_collector(node2), StatsCollector)
        assert isinstance(graph.get_in_stats_collector(node3), NoStatsCollector)
        assert mc.intermediate_output_tensors == [node1]
        assert mc.model_outputs == [node3]
        assert len(mc.stats_containers_list) == 2

    def test_bias_correction_creates_tensor2node(self, monkeypatch, fw_impl_mock, fw_info_mock):
        """
        Verify that when weights bias correction is enabled and a node has kernel weights to quantize,
        create_tensor2node is invoked for each incoming node.
        """
        # Set up nodes with quantization configurations for both activations and weights.
        node1 = build_node('node1', output_shape=(None, 3, 14), qcs=[build_qc(4), build_qc(2)])
        node1.is_activation_quantization_enabled = Mock(return_value=True)
        node1.is_weights_quantization_enabled = Mock(return_value=True)
        node2 = build_node('node2', output_shape=(None, 2, 71), qcs=[build_qc(4), build_qc(2)])
        node2.is_activation_quantization_enabled = Mock(return_value=True)
        node2.is_weights_quantization_enabled = Mock(return_value=True)
        node3 = build_node('node3', output_shape=(None, 59), qcs=[build_qc(4), build_qc(2)])
        node3.is_activation_quantization_enabled = Mock(return_value=True)
        node3.is_weights_quantization_enabled = Mock(return_value=False)

        # Build a graph connecting the nodes.
        graph = Graph('g',
                      input_nodes=[node1],
                      nodes=[node1, node2, node3],
                      output_nodes=[OutTensor(node3, 0)],
                      edge_list=[Edge(node1, node2, 0, 0), Edge(node2, node3, 0, 0)])
        graph.set_out_stats_collector_to_node = Mock(wraps=graph.set_out_stats_collector_to_node)

        # Return a kernel attribute to trigger bias correction.
        fw_info_mock.get_kernel_op_attributes.return_value = ['kernel']

        qc = DEFAULTCONFIG

        calls = []
        # Define a fake function to record call arguments for create_tensor2node.
        def fake_create_tensor2node(graph, node, fw_info):
            calls.append((graph, node, fw_info))

        # Patch create_tensor2node in the model_collector module.
        monkeypatch.setattr(model_collector, "create_tensor2node", fake_create_tensor2node)

        ModelCollector(graph, fw_impl_mock, fw_info_mock, qc=qc)

        fw_impl_mock.model_builder.assert_called_once()
        assert len(calls) == 1


class TestModelCollectorInfer:
    @pytest.fixture(autouse=True)
    def setup(self, fw_impl_mock, fw_info_mock):
        """
        Fixture to set up a graph with three nodes, fake model inference outputs,
        and a fake Hessian service for subsequent inference tests.
        """
        input_shape = (1, 3, 14)
        self.node1 = build_node('node1', output_shape=input_shape, qcs=[build_qc(4), build_qc(2)])
        self.node1.is_activation_quantization_enabled = Mock(return_value=True)
        self.node1.is_weights_quantization_enabled = Mock(return_value=True)
        self.node2 = build_node('node2', output_shape=input_shape, qcs=[build_qc(4), build_qc(2)])
        self.node2.is_activation_quantization_enabled = Mock(return_value=True)
        self.node2.is_weights_quantization_enabled = Mock(return_value=True)
        self.node3 = build_node('node3', output_shape=input_shape, qcs=[build_qc(4), build_qc(2)])
        self.node3.is_activation_quantization_enabled = Mock(return_value=True)
        self.node3.is_weights_quantization_enabled = Mock(return_value=False)

        self.graph = Graph('g',
                           input_nodes=[self.node1],
                           nodes=[self.node1, self.node2, self.node3],
                           output_nodes=[OutTensor(self.node3, 0)],
                           edge_list=[Edge(self.node1, self.node2, 0, 0), Edge(self.node2, self.node3, 0, 0)])

        # Simulate model inference outputs.
        self.fake_output1 = np.random.randn(*input_shape)
        self.fake_output2 = np.random.randn(*input_shape)
        self.fake_output3 = np.random.randn(*input_shape)
        fw_impl_mock.run_model_inference.return_value = [self.fake_output1, self.fake_output2, self.fake_output3]
        fw_impl_mock.to_numpy.side_effect = lambda x: x

        fw_info_mock.get_kernel_op_attributes.return_value = [None]

        # Set up a fake Hessian service with a predetermined hessian result.
        self.fake_hessian_info_service = Mock(spec_set=HessianInfoService)
        self.fake_hessian_result = np.random.randn(*input_shape)
        self.fake_hessian_info_service.fetch_hessian = Mock(
            wraps=self.fake_hessian_info_service.fetch_hessian,
            return_value={'node1': self.fake_hessian_result, 'node2': self.fake_hessian_result}
        )

        self.qc = DEFAULTCONFIG
        self.infer_input = [np.random.randn(*input_shape)]

    def test_infer_without_hessian(self, fw_impl_mock, fw_info_mock):
        """
        Verify that ModelCollector.infer calls run_model_inference without fetching hessian data
        when activation_error_method is not HMSE.
        """
        self.qc.activation_error_method = QuantizationErrorMethod.MSE
        mc = ModelCollector(self.graph, fw_impl_mock, fw_info_mock, qc=self.qc, hessian_info_service=self.fake_hessian_info_service)

        mc.infer(self.infer_input)
        # Check that inference runs without requiring gradients.
        fw_impl_mock.run_model_inference.assert_called_once_with(mc.model, self.infer_input, requires_grad=False)
        # Confirm that the Hessian service is not used.
        mc.hessian_service.fetch_hessian.assert_not_called()

    def test_infer_with_hessian(self, fw_impl_mock, fw_info_mock):
        """
        Verify that ModelCollector.infer fetches hessian data when activation_error_method is HMSE.
        """
        self.qc.activation_error_method = QuantizationErrorMethod.HMSE
        mc = ModelCollector(self.graph, fw_impl_mock, fw_info_mock, qc=self.qc, hessian_info_service=self.fake_hessian_info_service)

        mc.infer(self.infer_input)
        # Check that inference runs with gradients enabled.
        fw_impl_mock.run_model_inference.assert_called_once_with(mc.model, self.infer_input, requires_grad=True)
        # Confirm that the Hessian data is fetched.
        mc.hessian_service.fetch_hessian.assert_called_once()

    def test_update_statistics_called(self, fw_impl_mock, fw_info_mock):
        """
        Verify that update_statistics is called for each statistics container during inference.
        """
        # Create a dummy stats container that always requires collection.
        fake_stats_container = Mock(require_collection=lambda: True, update_statistics=Mock())
        self.graph.get_out_stats_collector = Mock(return_value=fake_stats_container)
        self.qc.activation_error_method = QuantizationErrorMethod.HMSE

        mc = ModelCollector(self.graph, fw_impl_mock, fw_info_mock, qc=self.qc, hessian_info_service=self.fake_hessian_info_service)
        mc.infer(self.infer_input)
        # Check that update_statistics is called with the corresponding outputs and hessian data.
        calls = fake_stats_container.update_statistics.call_args_list

        assert_array_equal(calls[0][0][0], self.fake_output1)
        assert_array_equal(calls[0][0][1], np.abs(self.fake_hessian_result))

        assert_array_equal(calls[1][0][0], self.fake_output2)
        assert_array_equal(calls[1][0][1], np.abs(self.fake_hessian_result))

        assert calls[2][0][0] is self.fake_output3
        assert calls[2][0][1] is None
