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

import keras
import unittest

from packaging import version
import tensorflow as tf

from model_compression_toolkit.core.common.framework_info import set_fw_info
from model_compression_toolkit.core.keras.default_framework_info import KerasInfo
from model_compression_toolkit.core.common.fusion.fusing_info import FusingInfoGenerator
from model_compression_toolkit.core.common.quantization.quantization_config import CustomOpsetLayers
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2keras import \
    AttachTpcToKeras
from tests.common_tests.helpers.generate_test_tpc import generate_test_op_qc, generate_test_attr_configs

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, Dense, BatchNormalization, ReLU, Input, Add, InputLayer
else:
    from keras.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, Dense, BatchNormalization, ReLU, Input, Add, InputLayer
import numpy as np

from model_compression_toolkit.core.common.graph.virtual_activation_weights_node import VirtualSplitActivationNode, \
    VirtualActivationWeightsNode, VirtualSplitWeightsNode
from model_compression_toolkit.core.common.quantization.filter_nodes_candidates import filter_nodes_candidates
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.virtual_activation_weights_composition import \
    VirtualActivationWeightsComposition
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.weights_activation_split import \
    WeightsActivationSplit
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs

import model_compression_toolkit as mct
from tests.keras_tests.tpc_keras import get_tpc_with_activation_mp_keras


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
    yield [np.random.randn(1, 8, 8, 3).astype(np.float32)]


def prepare_graph(in_model, keras_impl, mixed_precision_candidates_list, base_config, default_config):
    qc = mct.core.QuantizationConfig(custom_tpc_opset_to_layer={"Input": CustomOpsetLayers([InputLayer])})

    graph = keras_impl.model_reader(in_model, representative_dataset)  # model reading

    tpc = get_tpc_with_activation_mp_keras(base_config=base_config,
                                           default_config=default_config,
                                           mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                           name="activation_weights_composition_test")

    attach2keras = AttachTpcToKeras()
    fqc = attach2keras.attach(tpc, qc.custom_tpc_opset_to_layer)

    graph.set_fqc(fqc)

    # Standard graph substitutions
    graph = substitute(graph, keras_impl.get_substitutions_prepare_graph())
    for node in graph.nodes:
        node.prior_info = keras_impl.get_node_prior_info(node=node, graph=graph)
    graph = substitute(graph, keras_impl.get_substitutions_pre_statistics_collection(qc))

    graph = set_quantization_configuration_to_graph(graph=graph,
                                                    quant_config=qc,
                                                    mixed_precision_enable=True)

    fusing_info = FusingInfoGenerator(fqc.get_fusing_patterns()).generate_fusing_info(graph)
    graph.fusing_info = fusing_info
    graph.disable_fused_nodes_activation_quantization()

    graph = filter_nodes_candidates(graph)

    return graph


class TestActivationWeightsComposition(unittest.TestCase):
    def setUp(self):
        set_fw_info(KerasInfo)

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

    def test_compose_without_split_error(self):
        in_model = two_conv_model()
        keras_impl = KerasImplementation()

        base_config, _, default_config = get_op_quantization_configs()
        graph = prepare_graph(in_model, keras_impl,
                              mixed_precision_candidates_list=_get_base_mp_nbits_candidates(), base_config=base_config,
                              default_config=default_config)

        with self.assertRaises(TypeError) as e:
            substitute(copy.deepcopy(graph), [VirtualActivationWeightsComposition()])
        self.assertTrue('expected to be of type VirtualSplitWeightsNode' in str(e.exception))

    def test_two_conv_net_compose_after_split(self):
        in_model = two_conv_model()
        keras_impl = KerasImplementation()

        base_config, _, default_config = get_op_quantization_configs()
        graph = prepare_graph(in_model, keras_impl,
                              mixed_precision_candidates_list=_get_base_mp_nbits_candidates(), base_config=base_config,
                              default_config=default_config)

        # Validation is skipped because fusing information is not relevant for the virtual graph.
        # Therefore, validation checks are disabled before the virtual graph substitution and
        # re-enabled once it completes.
        graph.skip_validation_check = True

        # Nodes split and composition substitution
        split_graph = substitute(copy.deepcopy(graph), [WeightsActivationSplit()])
        v_graph = substitute(copy.deepcopy(split_graph), [VirtualActivationWeightsComposition()])

        graph.skip_validation_check = False

        self._verify_two_conv_with_split_test(graph, v_graph, 9, 9)

    def test_two_conv_net_compose_after_split_weights_only(self):
        in_model = two_conv_model()
        keras_impl = KerasImplementation()

        base_config, _, default_config = get_op_quantization_configs()
        base_config = base_config.clone_and_edit(enable_activation_quantization=False)
        graph = prepare_graph(in_model, keras_impl,
                              mixed_precision_candidates_list=_get_base_mp_nbits_candidates(), base_config=base_config,
                              default_config=default_config)

        # Validation is skipped because fusing information is not relevant for the virtual graph.
        # Therefore, validation checks are disabled before the virtual graph substitution and
        # re-enabled once it completes.
        graph.skip_validation_check = True

        # Nodes split and composition substitution
        split_graph = substitute(copy.deepcopy(graph), [WeightsActivationSplit()])
        v_graph = substitute(copy.deepcopy(split_graph), [VirtualActivationWeightsComposition()])

        graph.skip_validation_check = False

        self._verify_two_conv_with_split_test(graph, v_graph, 3, 3)

    def test_two_conv_net_compose_after_split_activation_only(self):
        in_model = two_conv_model()
        keras_impl = KerasImplementation()

        base_config = generate_test_op_qc(**generate_test_attr_configs(enable_kernel_weights_quantization=False))
        default_config = base_config.clone_and_edit(attr_weights_configs_mapping={})

        graph = prepare_graph(in_model, keras_impl,
                              mixed_precision_candidates_list=_get_base_mp_nbits_candidates(), base_config=base_config,
                              default_config=default_config)

        # Validation is skipped because fusing information is not relevant for the virtual graph.
        # Therefore, validation checks are disabled before the virtual graph substitution and
        # re-enabled once it completes.
        graph.skip_validation_check = True

        # Nodes split and composition substitution
        split_graph = substitute(copy.deepcopy(graph), [WeightsActivationSplit()])
        v_graph = substitute(copy.deepcopy(split_graph), [VirtualActivationWeightsComposition()])

        graph.skip_validation_check = False

        self._verify_two_conv_with_split_test(graph, v_graph, 3, 3)

    def test_all_weights_layers_composition(self):
        in_model = multiple_weights_nodes_model()
        keras_impl = KerasImplementation()

        base_config, _, default_config = get_op_quantization_configs()
        graph = prepare_graph(in_model, keras_impl,
                              mixed_precision_candidates_list=_get_base_mp_nbits_candidates(),
                              base_config=base_config,
                              default_config=default_config)

        # Validation is skipped because fusing information is not relevant for the virtual graph.
        # Therefore, validation checks are disabled before the virtual graph substitution and
        # re-enabled once it completes.
        graph.skip_validation_check = True

        # Nodes split and composition substitution
        split_graph = substitute(copy.deepcopy(graph), [WeightsActivationSplit()])
        v_graph = substitute(copy.deepcopy(split_graph), [VirtualActivationWeightsComposition()])

        graph.skip_validation_check = False

        assert split_graph is not graph
        self.assertTrue(len(v_graph.nodes) == 8)
        self.assertTrue(len([n for n in v_graph.nodes if isinstance(n, VirtualActivationWeightsNode)]) == 5)

        sorted_v_nodes = v_graph.get_topo_sorted_nodes()
        # Input-Conv1 composed node
        self.assertTrue(len(sorted_v_nodes[0].candidates_quantization_cfg) == 9)
        # Conv1-Conv2 composed node
        self.assertTrue(len(sorted_v_nodes[1].candidates_quantization_cfg) == 9)
        # Conv2-Activation node
        self.assertTrue(isinstance(sorted_v_nodes[2], VirtualSplitActivationNode))
        self.assertTrue(len(sorted_v_nodes[2].candidates_quantization_cfg) == 1)
        # Relu1-ConvTranspose composed node
        self.assertTrue(len(sorted_v_nodes[3].candidates_quantization_cfg) == 9)
        # ConvTranspose Activation
        self.assertTrue(isinstance(sorted_v_nodes[4], VirtualSplitActivationNode))
        self.assertTrue(len(sorted_v_nodes[4].candidates_quantization_cfg) == 1)
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

        base_config, _, default_config = get_op_quantization_configs()
        graph = prepare_graph(in_model, keras_impl,
                              mixed_precision_candidates_list=_get_base_mp_nbits_candidates(), base_config=base_config,
                              default_config=default_config)

        # Validation is skipped because fusing information is not relevant for the virtual graph.
        # Therefore, validation checks are disabled before the virtual graph substitution and
        # re-enabled once it completes.
        graph.skip_validation_check = True

        split_graph = substitute(copy.deepcopy(graph), [WeightsActivationSplit()])
        v_graph = substitute(copy.deepcopy(split_graph), [VirtualActivationWeightsComposition()])

        graph.skip_validation_check = False

        #  Since the only activation before the convolutions is the Input layer activation, and it goes to both
        # convolutions (the input node has multiple output edges) no composition should be made.
        self.assertTrue(len(v_graph.nodes) == len(split_graph.nodes))
        self.assertTrue(not any([isinstance(n, VirtualActivationWeightsNode) for n in v_graph.nodes]))
        virtual_activations = [n for n in v_graph.nodes if isinstance(n, VirtualSplitActivationNode)]
        virtual_weights = [n for n in v_graph.nodes if isinstance(n, VirtualSplitWeightsNode)]
        self.assertEqual(len(virtual_activations), 2)
        self.assertEqual(len(virtual_weights), 2)

        sorted_v_nodes = v_graph.get_topo_sorted_nodes()
        for i, n in enumerate(split_graph.get_topo_sorted_nodes()):
            self.assertTrue(n.name == sorted_v_nodes[i].name)

    def test_activation_with_const(self):
        inputs = Input(shape=INPUT_SHAPE)
        x = tf.add(inputs, np.ones(INPUT_SHAPE[-1]))
        x = Conv2D(filters=2, kernel_size=3)(x)
        model = keras.Model(inputs=inputs, outputs=x)

        keras_impl = KerasImplementation()

        base_config, _, default_config = get_op_quantization_configs()
        graph = prepare_graph(model, keras_impl,
                              mixed_precision_candidates_list=_get_base_mp_nbits_candidates(), base_config=base_config,
                              default_config=default_config)

        split_graph = substitute(copy.deepcopy(graph), [WeightsActivationSplit()])
        v_graph = substitute(copy.deepcopy(split_graph), [VirtualActivationWeightsComposition()])

        nodes = v_graph.get_topo_sorted_nodes()
        self.assertTrue(len(nodes) == 3)
        self.assertTrue(isinstance(nodes[1], VirtualActivationWeightsNode))
        aw_node = nodes[1]
        self.assertTrue(len(aw_node.candidates_quantization_cfg) == 9)

        orig_add, orig_conv = graph.get_topo_sorted_nodes()[1:]
        self.assertFalse(orig_add.has_any_configurable_weight())
        pos_weights_cfg = orig_add.candidates_quantization_cfg[0].weights_quantization_cfg.pos_attributes_config_mapping
        self.assertTrue(all(c.weights_quantization_cfg.pos_attributes_config_mapping == pos_weights_cfg
                            for c in aw_node.candidates_quantization_cfg))
        self.assertTrue(len(aw_node.weights) == 3)

        self.assertTrue(np.array_equal(aw_node.weights[1], orig_add.weights[1]))
        self.assertTrue(all(np.array_equal(aw_node.weights[k], v) for k, v in orig_conv.weights.items()))
