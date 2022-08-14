# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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

import keras
import unittest

from keras.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, Dense, BatchNormalization, ReLU, Input
import numpy as np

from model_compression_toolkit import DEFAULTCONFIG, MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.common.fusion.layer_fusing import fusion
from model_compression_toolkit.core.common.graph.virtual_activation_weights_node import VirtualSplitActivationNode, \
    VirtualSplitWeightsNode
from model_compression_toolkit.core.common.quantization.filter_nodes_candidates import filter_nodes_candidates
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.weight_activation_split import \
    WeightsActivationSplit
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs

import model_compression_toolkit as mct
from tests.common_tests.helpers.activation_mp_tp_model import generate_tp_model_with_activation_mp
from tests.keras_tests.tpc_keras import generate_activation_mp_tpc_keras

tp = mct.target_platform

INPUT_SHAPE = (8, 8, 3)


def _get_base_mp_nbits_candidates():
    return [(4, 8), (4, 4), (4, 2),
            (8, 8), (8, 4), (8, 2),
            (2, 8), (2, 4), (2, 2)]


def single_conv_model():
    inputs = Input(shape=INPUT_SHAPE)
    x = Conv2D(2, 3)(inputs)
    x = BatchNormalization()(x)
    outputs = ReLU()(x)
    return keras.Model(inputs=inputs, outputs=outputs)


def multiple_weights_nodes_model():
    inputs = Input(shape=INPUT_SHAPE)
    x = Conv2D(2, 3)(inputs)
    x = Conv2DTranspose(4, 3)(x)
    x = DepthwiseConv2D(3, depth_multiplier=10)(x)
    outputs = Dense(20)(x)
    return keras.Model(inputs=inputs, outputs=outputs)


def representative_dataset():
    return [np.random.randn(1, 8, 8, 3).astype(np.float32)]


def prepare_graph(in_model, keras_impl, mixed_precision_candidates_list):
    fw_info = DEFAULT_KERAS_INFO
    qc = MixedPrecisionQuantizationConfig(DEFAULTCONFIG)

    graph = keras_impl.model_reader(in_model, representative_dataset)  # model reading

    base_config, _ = get_op_quantization_configs()
    mp_tp_model = generate_tp_model_with_activation_mp(base_config, mixed_precision_candidates_list)
    tpc = generate_activation_mp_tpc_keras(tp_model=mp_tp_model)

    graph.set_fw_info(fw_info)
    graph.set_tpc(tpc)

    # Standard graph substitutions
    graph = substitute(graph, keras_impl.get_substitutions_prepare_graph())
    for node in graph.nodes:
        node.prior_info = keras_impl.get_node_prior_info(node=node,
                                                         fw_info=fw_info, graph=graph)
    graph = substitute(graph, keras_impl.get_substitutions_pre_statistics_collection(qc))

    graph = set_quantization_configuration_to_graph(graph=graph,
                                                    quant_config=qc,
                                                    mixed_precision_enable=True)
    graph = fusion(graph, tpc)
    graph = filter_nodes_candidates(graph)

    # Split graph substitution
    split_graph = substitute(graph, [WeightsActivationSplit()])

    return graph, split_graph


class TestWeightsActivationSplit(unittest.TestCase):

    def _verify_single_conv_test(self, graph, split_graph, num_weights_candidates, num_activation_candidates):
        # verify that the convolution layer was split and that the new virtual node have the correct candidates
        self.assertTrue(len(split_graph.nodes) == len(graph.nodes) + 1,
                        "Split graph should have exactly 1 more node than the original graph.")

        split_weights_node = split_graph.get_topo_sorted_nodes()[1]
        split_activation_node = split_graph.get_topo_sorted_nodes()[2]
        self.assertTrue(isinstance(split_weights_node, VirtualSplitWeightsNode))
        self.assertTrue(isinstance(split_activation_node, VirtualSplitActivationNode))

        self.assertTrue(len(split_weights_node.candidates_quantization_cfg) == num_weights_candidates,
                        "The weights split node should have only weights configurable candidates.")
        self.assertTrue(len(split_activation_node.candidates_quantization_cfg) == num_activation_candidates,
                        "The activation split node should have only activation configurable candidates.")

        self.assertTrue(not any([c.activation_quantization_cfg.enable_activation_quantization
                                 for c in split_weights_node.candidates_quantization_cfg]),
                        "All weights node's candidates activation quantization should be disabled.")
        self.assertTrue(not any([c.weights_quantization_cfg.enable_weights_quantization
                                 for c in split_activation_node.candidates_quantization_cfg]),
                        "All activation node's candidates weights quantization should be disabled.")

        origin_conv = graph.get_topo_sorted_nodes()[1]
        self.assertTrue(split_weights_node.origin_node.name == origin_conv.name)
        self.assertTrue(split_activation_node.origin_node.name == origin_conv.name)
        self.assertTrue(split_graph.out_edges(split_weights_node)[0].sink_node == split_activation_node)

    def test_single_conv_net_split(self):
        in_model = single_conv_model()
        keras_impl = KerasImplementation()

        graph, split_graph = prepare_graph(in_model, keras_impl, mixed_precision_candidates_list=_get_base_mp_nbits_candidates())
        self._verify_single_conv_test(graph, split_graph, num_weights_candidates=3, num_activation_candidates=3)

    def test_single_conv_net_weights_only_split(self):
        in_model = single_conv_model()
        keras_impl = KerasImplementation()

        graph, split_graph = prepare_graph(in_model, keras_impl, mixed_precision_candidates_list=[(8, 8), (4, 8), (2, 8)])
        self._verify_single_conv_test(graph, split_graph, num_weights_candidates=3, num_activation_candidates=1)

    def test_single_conv_net_activation_only_split(self):
        in_model = single_conv_model()
        keras_impl = KerasImplementation()

        graph, split_graph = prepare_graph(in_model, keras_impl, mixed_precision_candidates_list=[(8, 8), (8, 4), (8, 2)])
        self._verify_single_conv_test(graph, split_graph, num_weights_candidates=1, num_activation_candidates=3)

    def test_all_weights_layers_split(self):
        in_model = multiple_weights_nodes_model()
        keras_impl = KerasImplementation()

        graph, split_graph = prepare_graph(in_model, keras_impl, mixed_precision_candidates_list=_get_base_mp_nbits_candidates())
        weights_node_types = [Conv2D, Conv2DTranspose, DepthwiseConv2D, Dense]
        original_weights_nodes = [n for n in graph.get_topo_sorted_nodes() if n.type in weights_node_types]

        self.assertTrue(len(split_graph.nodes) == len(graph.nodes) + len(original_weights_nodes))

        for n in split_graph.get_topo_sorted_nodes():
            if isinstance(n, VirtualSplitWeightsNode):
                self.assertTrue(len(n.candidates_quantization_cfg) == 3,
                                "The weights split node should have only weights configurable candidates.")
                self.assertTrue(not any([c.activation_quantization_cfg.enable_activation_quantization
                                         for c in n.candidates_quantization_cfg]),
                                "All weights node's candidates activation quantization should be disabled.")
            if isinstance(n, VirtualSplitActivationNode):
                self.assertTrue(len(n.candidates_quantization_cfg) == 3,
                                "The activation split node should have only activation configurable candidates.")
                self.assertTrue(not any([c.weights_quantization_cfg.enable_weights_quantization
                                         for c in n.candidates_quantization_cfg]),
                                "All activation node's candidates weights quantization should be disabled.")

    def test_non_composite_candidates_config(self):
        in_model = single_conv_model()
        keras_impl = KerasImplementation()

        with self.assertRaises(Exception):
            prepare_graph(in_model, keras_impl, mixed_precision_candidates_list=[(8, 2), (2, 4), (4, 8)])
