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

from keras.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, Dense, BatchNormalization, ReLU, Input, Add
import numpy as np

from model_compression_toolkit import DEFAULTCONFIG, MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.common.fusion.layer_fusing import fusion
from model_compression_toolkit.core.common.graph.virtual_activation_weights_node import VirtualSplitActivationNode, \
    VirtualSplitWeightsNode, VirtualActivationWeightsNode
from model_compression_toolkit.core.common.quantization.filter_nodes_candidates import filter_nodes_candidates
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.virtual_activation_weights_composition import \
    VirtualActivationWeightsComposition
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


def two_conv_model():
    inputs = Input(shape=INPUT_SHAPE)
    x = Conv2D(2, 3)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    outputs = Conv2D(2, 3)(x)
    return keras.Model(inputs=inputs, outputs=outputs)


def multiple_weights_nodes_model():
    inputs = Input(shape=INPUT_SHAPE)
    x = Conv2D(2, 3)(inputs)
    x = Conv2D(2, 3)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(4, 3)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = DepthwiseConv2D(3, depth_multiplier=10)(x)
    outputs = Dense(20)(x)
    return keras.Model(inputs=inputs, outputs=outputs)


def multiple_outputs_activation_model():
    inputs = Input(shape=INPUT_SHAPE)
    x = Conv2D(2, 3)(inputs)
    y = Conv2D(2, 3)(inputs)
    x_relu = ReLU()(x)
    y_relu = ReLU()(y)
    outputs = Add()([x_relu, y_relu])
    return keras.Model(inputs=inputs, outputs=outputs)


def representative_dataset():
    return [np.random.randn(1, 8, 8, 3).astype(np.float32)]


def prepare_graph(in_model, keras_impl, mixed_precision_candidates_list, base_config):
    fw_info = DEFAULT_KERAS_INFO
    qc = MixedPrecisionQuantizationConfig(DEFAULTCONFIG)

    graph = keras_impl.model_reader(in_model, representative_dataset)  # model reading

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

    return graph


class TestActivationWeightsComposition(unittest.TestCase):

    def _verify_two_conv_with_split_test(self, graph, v_graph, num_weights_candidates, num_activation_candidates):
        self.assertTrue(len(v_graph.nodes) == len(graph.nodes),
                        "Both convolutions should be split and then composed with their predecessor activation node."
                        "So no change in number of node is expected.")
        self.assertTrue(len(graph.input_nodes) == len(v_graph.input_nodes))
        self.assertTrue(len(graph.output_nodes) == len(v_graph.output_nodes))

        composed_node_1 = v_graph.get_topo_sorted_nodes()[0]
        composed_node_2 = v_graph.get_topo_sorted_nodes()[2]

        self.assertTrue(len(composed_node_1.candidates_quantization_cfg) == num_weights_candidates,
                        "The composed node should have the cartesian product of the activation and weights nodes candidates.")
        self.assertTrue(len(composed_node_2.candidates_quantization_cfg) == num_activation_candidates,
                        "The composed node should have the cartesian product of the activation and weights nodes candidates.")

    def test_two_conv_net_compose_without_split(self):
        """
        Note that this test checks a hypothetical case that should not be used - we should not run nodes
        composition without running weights nodes split before, otherwise we'll might get duplication
        in the composed node's candidates.
        """

        in_model = two_conv_model()
        keras_impl = KerasImplementation()

        base_config, _ = get_op_quantization_configs()
        graph = prepare_graph(in_model, keras_impl,
                              mixed_precision_candidates_list=_get_base_mp_nbits_candidates(), base_config=base_config)

        # Nodes composition substitution
        v_graph = substitute(graph, [VirtualActivationWeightsComposition()])

        self.assertTrue(len(v_graph.nodes) == len(graph.nodes) - 2,
                        "Both convolution nodes should be composed with their predecessor activation node.")
        self.assertTrue(len(graph.input_nodes) == len(v_graph.input_nodes))
        self.assertTrue(len(graph.output_nodes) == len(v_graph.output_nodes))

        composed_node_1 = v_graph.get_topo_sorted_nodes()[0]
        composed_node_2 = v_graph.get_topo_sorted_nodes()[1]
        graph_sorted_nodes = graph.get_topo_sorted_nodes()

        self.assertTrue(composed_node_1.original_activation_node.name == graph_sorted_nodes[0].name)
        self.assertTrue(composed_node_1.original_weights_node.name == graph_sorted_nodes[1].name)
        self.assertTrue(composed_node_2.original_activation_node.name == graph_sorted_nodes[2].name)
        self.assertTrue(composed_node_2.original_weights_node.name == graph_sorted_nodes[3].name)

    def test_two_conv_net_compose_after_split(self):
        in_model = two_conv_model()
        keras_impl = KerasImplementation()

        base_config, _ = get_op_quantization_configs()
        graph = prepare_graph(in_model, keras_impl,
                              mixed_precision_candidates_list=_get_base_mp_nbits_candidates(), base_config=base_config)

        # Nodes split and composition substitution
        split_graph = substitute(graph, [WeightsActivationSplit()])
        v_graph = substitute(split_graph, [VirtualActivationWeightsComposition()])

        self._verify_two_conv_with_split_test(graph, v_graph, 9, 9)

    def test_two_conv_net_compose_after_split_weights_only(self):
        in_model = two_conv_model()
        keras_impl = KerasImplementation()

        base_config, _ = get_op_quantization_configs()
        base_config = base_config.clone_and_edit(enable_activation_quantization=False)
        graph = prepare_graph(in_model, keras_impl,
                              mixed_precision_candidates_list=_get_base_mp_nbits_candidates(), base_config=base_config)

        # Nodes split and composition substitution
        split_graph = substitute(graph, [WeightsActivationSplit()])
        v_graph = substitute(split_graph, [VirtualActivationWeightsComposition()])

        self._verify_two_conv_with_split_test(graph, v_graph, 3, 3)

    def test_two_conv_net_compose_after_split_activation_only(self):
        in_model = two_conv_model()
        keras_impl = KerasImplementation()

        base_config, _ = get_op_quantization_configs()
        base_config = base_config.clone_and_edit(enable_weights_quantization=False)
        graph = prepare_graph(in_model, keras_impl,
                              mixed_precision_candidates_list=_get_base_mp_nbits_candidates(), base_config=base_config)

        # Nodes split and composition substitution
        split_graph = substitute(graph, [WeightsActivationSplit()])
        v_graph = substitute(split_graph, [VirtualActivationWeightsComposition()])

        self._verify_two_conv_with_split_test(graph, v_graph, 3, 3)

    def test_all_weights_layers_composition(self):
        in_model = multiple_weights_nodes_model()
        keras_impl = KerasImplementation()

        base_config, _ = get_op_quantization_configs()
        graph = prepare_graph(in_model, keras_impl,
                              mixed_precision_candidates_list=_get_base_mp_nbits_candidates(),
                              base_config=base_config)

        # Nodes split and composition substitution
        split_graph = substitute(graph, [WeightsActivationSplit()])
        v_graph = substitute(split_graph, [VirtualActivationWeightsComposition()])

        self.assertTrue(len(v_graph.nodes) == 8)
        self.assertTrue(len([n for n in v_graph.nodes if isinstance(n, VirtualActivationWeightsNode)]) == 5)

        sorted_v_nodes = v_graph.get_topo_sorted_nodes()
        # Input-Conv1 composed node
        self.assertTrue(len(sorted_v_nodes[0].candidates_quantization_cfg) == 9)
        # Conv1-Conv2 composed node
        self.assertTrue(len(sorted_v_nodes[1].candidates_quantization_cfg) == 9)
        # Conv2-Activation node
        self.assertTrue(isinstance(sorted_v_nodes[2], VirtualSplitActivationNode))
        self.assertTrue(len(sorted_v_nodes[2].candidates_quantization_cfg) == 3)
        # Relu1-ConvTranspose composed node
        self.assertTrue(len(sorted_v_nodes[3].candidates_quantization_cfg) == 9)
        # ConvTranspose Activation
        self.assertTrue(isinstance(sorted_v_nodes[4], VirtualSplitActivationNode))
        self.assertTrue(len(sorted_v_nodes[4].candidates_quantization_cfg) == 3)
        # Relu2-Depthwise composed node
        self.assertTrue(len(sorted_v_nodes[5].candidates_quantization_cfg) == 9)
        # Depthwise-Dense composed node
        self.assertTrue(len(sorted_v_nodes[6].candidates_quantization_cfg) == 9)
        # Dense Activation
        self.assertTrue(isinstance(sorted_v_nodes[7], VirtualSplitActivationNode))
        self.assertTrue(len(sorted_v_nodes[7].candidates_quantization_cfg) == 3)

    def test_multiple_output_activation(self):
        in_model = multiple_outputs_activation_model()
        keras_impl = KerasImplementation()

        base_config, _ = get_op_quantization_configs()
        graph = prepare_graph(in_model, keras_impl,
                              mixed_precision_candidates_list=_get_base_mp_nbits_candidates(), base_config=base_config)

        # Nodes composition substitution
        v_graph = substitute(graph, [VirtualActivationWeightsComposition()])

        #  Since the only activation before the convolutions is the Input layer activation, and it goes to both
        # convolutions (the input node has multiple output edges) no composition should be made.
        self.assertTrue(len(v_graph.nodes) == len(graph.nodes))
        self.assertTrue(not any([isinstance(n, VirtualActivationWeightsNode) for n in v_graph.nodes]))

        sorted_v_nodes = v_graph.get_topo_sorted_nodes()
        for i, n in enumerate(graph.get_topo_sorted_nodes()):
            self.assertTrue(n.name == sorted_v_nodes[i].name)
