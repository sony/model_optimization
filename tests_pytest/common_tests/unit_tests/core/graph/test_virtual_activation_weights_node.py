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
import numpy as np
import pytest

from model_compression_toolkit.core.common.framework_info import set_fw_info
from model_compression_toolkit.core.common.graph.virtual_activation_weights_node import VirtualActivationWeightsNode
from tests_pytest._test_util.graph_builder_utils import build_node, DummyLayer, build_nbits_qc


class DummyLayerWKernel:
    pass


class TestVirtualActivationWeightsNode:
    @pytest.fixture(autouse=True)
    def setup(self, fw_info_mock):
        set_fw_info(fw_info_mock)

    # TODO tests only cover combining weights from activation and weight nodes and errors.
    def test_activation_with_weights(self, fw_info_mock):
        """ Tests that weights from activation and weight node are combined correctly. """
        # Each node has a unique weight attr and a unique positional weights. In addition, both nodes have
        # an identical canonical attribute (but different full name), and an identical positional weight.
        # All weights have different quantization.
        fw_info_mock.get_kernel_op_attribute = lambda nt: 'weight' if nt is DummyLayerWKernel else None

        a_node = build_node('a', final_weights={'aaweightaa': np.ones((3, 14)), 'foo': np.ones(15),
                                                1: np.ones(15), 2: np.ones((5, 9))}, qcs=[build_nbits_qc(a_nbits=5,
                                                                                                         w_attr={
                                                                                                             'aaweightaa': (
                                                                                                                 2,
                                                                                                                 True),
                                                                                                             'foo': (
                                                                                                                 3,
                                                                                                                 True)},
                                                                                                         pos_attr=(
                                                                                                             4, True,
                                                                                                             [1, 2]),
                                                                                                         convert_canonical_attr=False)],
                            layer_class=DummyLayer)
        w_node = build_node('w', final_weights={'wwweightww': np.ones((2, 71)), 'bar': np.ones(8),
                                                1: np.ones(28), 3: np.ones(18)}, qcs=[build_nbits_qc(a_nbits=6,
                                                                                                     w_attr={
                                                                                                         'wwweightww': (
                                                                                                             5, True),
                                                                                                         'bar': (
                                                                                                             6, True)},
                                                                                                     pos_attr=(
                                                                                                         7, True,
                                                                                                         [1, 3]),
                                                                                                     convert_canonical_attr=False)],
                            layer_class=DummyLayerWKernel)

        v_node = VirtualActivationWeightsNode(a_node, w_node)
        assert len(v_node.weights) == 8

        assert len(w_node.weights) == len(a_node.weights) == 4
        # weights from weight node are unchanged
        for k, v in w_node.weights.items():
            assert np.array_equal(v_node.weights.pop(k), v)
        # unique weights from activation node are unchanged
        assert np.array_equal(v_node.weights.pop('foo'), a_node.weights['foo'])
        assert np.array_equal(v_node.weights.pop(2), a_node.weights[2])
        # duplicate positional weight
        assert np.array_equal(v_node.weights.pop(101), a_node.weights[1])
        # duplicate weight attribute
        [(new_attr, w)] = v_node.weights.items()
        assert 'weight' not in new_attr
        assert np.array_equal(w, a_node.weights['aaweightaa'])

        assert len(v_node.candidates_quantization_cfg) == 1
        v_qc = v_node.candidates_quantization_cfg[0]
        v_attr_cfg = v_qc.weights_quantization_cfg.attributes_config_mapping
        v_pos_cfg = v_qc.weights_quantization_cfg.pos_attributes_config_mapping
        a_qc = a_node.candidates_quantization_cfg[0]
        w_qc = w_node.candidates_quantization_cfg[0]

        assert v_attr_cfg == {
            'wwweightww': w_qc.weights_quantization_cfg.attributes_config_mapping['wwweightww'],
            'bar': w_qc.weights_quantization_cfg.attributes_config_mapping['bar'],
            'foo': a_qc.weights_quantization_cfg.attributes_config_mapping['foo'],
            new_attr: a_qc.weights_quantization_cfg.attributes_config_mapping['aaweightaa']
        }
        assert v_pos_cfg == {
            1: w_qc.weights_quantization_cfg.pos_attributes_config_mapping[1],
            101: a_qc.weights_quantization_cfg.pos_attributes_config_mapping[1],
            2: a_qc.weights_quantization_cfg.pos_attributes_config_mapping[2],
            3: w_qc.weights_quantization_cfg.pos_attributes_config_mapping[3]
        }

    def test_invalid_configurable_w_node_weight(self, fw_info_mock):
        fw_info_mock.get_kernel_op_attribute = lambda nt: 'kernel' if nt is DummyLayerWKernel else None

        w_node = build_node('w', canonical_weights={'kernel': np.ones(3), 'foo': np.ones(14)}, qcs=[
            build_nbits_qc(w_attr={'kernel': (8, True), 'foo': (8, True)}),
            build_nbits_qc(w_attr={'kernel': (8, True), 'foo': (4, True)})
        ], layer_class=DummyLayerWKernel)
        a_node = build_node('a', qcs=[build_nbits_qc()])

        with pytest.raises(NotImplementedError, match='Only kernel weight can be configurable. Got configurable .*foo'):
            VirtualActivationWeightsNode(a_node, w_node)

    def test_invalid_a_node_configurable_weight(self, fw_info_mock):
        fw_info_mock.get_kernel_op_attribute = lambda nt: 'kernel' if nt is DummyLayerWKernel else None

        w_node = build_node('w', canonical_weights={'kernel': np.ones(3), 'foo': np.ones(14)}, qcs=[
            build_nbits_qc(w_attr={'kernel': (8, True), 'foo': (8, True)}),
            build_nbits_qc(w_attr={'kernel': (4, True), 'foo': (8, True)})
        ], layer_class=DummyLayerWKernel)
        a_node = build_node('aaa', canonical_weights={'bar': np.ones(3), 'baz': np.ones(14)}, qcs=[
            build_nbits_qc(w_attr={'bar': (8, True), 'baz': (8, True)}),
            build_nbits_qc(w_attr={'bar': (8, True), 'baz': (4, True)})
        ])
        with pytest.raises(NotImplementedError, match='Node .*aaa with a configurable weight cannot be used as '
                                                      'activation for VirtualActivationWeightsNode'):
            VirtualActivationWeightsNode(a_node, w_node)

    def test_invalid_a_node_kernel(self, fw_info_mock):
        fw_info_mock.get_kernel_op_attribute = lambda nt: 'weight' if nt is DummyLayerWKernel else 'kernel'
        w_node = build_node('w', canonical_weights={'weight': np.ones(3)},
                            qcs=[build_nbits_qc(w_attr={'weight': (8, True)})], layer_class=DummyLayerWKernel)
        a_node = build_node('aaa', canonical_weights={'kernel': np.ones(3)},
                            qcs=[build_nbits_qc(w_attr={'kernel': (8, True)})])

        with pytest.raises(NotImplementedError, match='Node .*aaa with kernel cannot be used as '
                                                      'activation for VirtualActivationWeightsNode'):
            VirtualActivationWeightsNode(a_node, w_node)

