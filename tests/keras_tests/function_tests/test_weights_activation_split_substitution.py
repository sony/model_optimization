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
import copy
import tensorflow as tf
import keras
import unittest

from model_compression_toolkit.core.common.quantization.quantization_config import CustomOpsetLayers
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2keras import \
    AttachTpcToKeras

if tf.__version__ >= "2.13":
    from keras.src.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, Dense, BatchNormalization, ReLU, Input
    from keras.src.engine.input_layer import InputLayer
else:
    from keras.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, Dense, BatchNormalization, ReLU, Input
    from keras.engine.input_layer import InputLayer

import numpy as np

from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.core.common.graph.virtual_activation_weights_node import VirtualSplitActivationNode, \
    VirtualSplitWeightsNode
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.weights_activation_split import \
    WeightsActivationSplit
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs

import model_compression_toolkit as mct
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_configs
from tests.keras_tests.tpc_keras import get_tpc_with_activation_mp_keras


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
    yield [np.random.randn(1, 8, 8, 3).astype(np.float32)]


def get_tpc(mixed_precision_candidates_list):
    base_config, _, default_config = get_op_quantization_configs()

    return get_tpc_with_activation_mp_keras(base_config=base_config,
                                            default_config=default_config,
                                            mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                            name="weights_activation_split_test")


def setup_test(in_model, keras_impl, mixed_precision_candidates_list):
    graph = prepare_graph_with_configs(in_model, keras_impl, representative_dataset,
                                       lambda name, _tp: get_tpc(mixed_precision_candidates_list),
                                       mixed_precision_enabled=True,
                                       attach2fw=AttachTpcToKeras(),
                                       qc=QuantizationConfig(custom_tpc_opset_to_layer={"Input": CustomOpsetLayers([InputLayer])}))

    # Validation is skipped because fusing information is not relevant for the virtual graph.
    # Therefore, validation checks are disabled before the virtual graph substitution and
    # re-enabled once it completes.
    graph.skip_validation_check = True

    # Split graph substitution
    split_graph = substitute(copy.deepcopy(graph), [WeightsActivationSplit()])

    graph.skip_validation_check = False

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

        graph, split_graph = setup_test(in_model, keras_impl, mixed_precision_candidates_list=_get_base_mp_nbits_candidates())
        # num_activation_candidates here is 1 because the split Conv has ReLU after it - thenbecause of fusion, the
        # Conv layer doesn't have activation quantization candidates
        self._verify_single_conv_test(graph, split_graph, num_weights_candidates=3, num_activation_candidates=1)

    def test_single_conv_net_weights_only_split(self):
        in_model = single_conv_model()
        keras_impl = KerasImplementation()

        graph, split_graph = setup_test(in_model, keras_impl, mixed_precision_candidates_list=[(8, 8), (4, 8), (2, 8)])
        self._verify_single_conv_test(graph, split_graph, num_weights_candidates=3, num_activation_candidates=1)

    def test_single_conv_net_activation_only_split(self):
        in_model = single_conv_model()
        keras_impl = KerasImplementation()

        graph, split_graph = setup_test(in_model, keras_impl, mixed_precision_candidates_list=[(8, 8), (8, 4), (8, 2)])
        # num_activation_candidates here is 1 because the split Conv has ReLU after it - thenbecause of fusion, the
        # Conv layer doesn't have activation quantization candidates
        self._verify_single_conv_test(graph, split_graph, num_weights_candidates=1, num_activation_candidates=1)

    def test_all_weights_layers_split(self):
        in_model = multiple_weights_nodes_model()
        keras_impl = KerasImplementation()

        graph, split_graph = setup_test(in_model, keras_impl, mixed_precision_candidates_list=_get_base_mp_nbits_candidates())
        weights_node_types = [Conv2D, Conv2DTranspose, DepthwiseConv2D, Dense]
        original_weights_nodes = [n for n in graph.get_topo_sorted_nodes() if any([n.is_match_type(_type) for _type in weights_node_types])]

        self.assertTrue(len(split_graph.nodes) == len(graph.nodes) + len(original_weights_nodes))

        for n in split_graph.get_topo_sorted_nodes():
            if isinstance(n, VirtualSplitWeightsNode):
                self.assertTrue(len(n.candidates_quantization_cfg) == 3)
                self.assertFalse(any(c.activation_quantization_cfg.enable_activation_quantization
                                     for c in n.candidates_quantization_cfg))
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
            setup_test(in_model, keras_impl, mixed_precision_candidates_list=[(8, 2), (2, 4), (4, 8)])
