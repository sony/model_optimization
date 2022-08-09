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
from tensorflow.keras.layers import Conv2D, ReLU, Input

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.constants import DEFAULT_CANDIDATE_BITWIDTH
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    DEFAULT_MIXEDPRECISION_CONFIG
from model_compression_toolkit.core.common.quantization.filter_nodes_candidates import filter_nodes_candidates
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.core.common.fusion.layer_fusing import fusion
from tests.common_tests.helpers.activation_mp_tp_model import generate_tp_model_with_activation_mp
from tests.keras_tests.tpc_keras import generate_activation_mp_tpc_keras

tp = mct.target_platform


def get_base_config():
    return tp.OpQuantizationConfig(
        activation_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        weights_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        weights_n_bits=8,
        weights_per_channel_threshold=True,
        enable_weights_quantization=True,
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        weights_multiplier_nbits=None
    )


def get_full_bitwidth_candidates():
    return [(4, 8), (4, 4), (4, 2),
            (8, 8), (8, 4), (8, 2),
            (2, 8), (2, 4), (2, 2)]


def prepare_graph(in_model, base_config, bitwidth_candidates):
    tp = generate_tp_model_with_activation_mp(base_config, bitwidth_candidates)
    tpc = generate_activation_mp_tpc_keras(name="candidates_filter_test", tp_model=tp)

    fw_info = DEFAULT_KERAS_INFO
    keras_impl = KerasImplementation()
    graph = keras_impl.model_reader(in_model, None)  # model reading
    graph.set_tpc(tpc)
    graph.set_fw_info(fw_info)
    graph = set_quantization_configuration_to_graph(graph=graph,
                                                    quant_config=DEFAULT_MIXEDPRECISION_CONFIG,
                                                    mixed_precision_enable=True)
    graph = fusion(graph, tpc)

    return graph


def create_model_conv2d_only(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(2, 3)(inputs)
    outputs = Conv2D(2, 3)(x)
    return keras.Model(inputs=inputs, outputs=outputs)


def create_model_single_conv2d(input_shape):
    inputs = Input(shape=input_shape)
    outputs = Conv2D(2, 3)(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)


def create_model_conv2d_relu(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(2, 3)(inputs)
    outputs = ReLU()(x)
    return keras.Model(inputs=inputs, outputs=outputs)


class TestCfgCandidatesFilter(unittest.TestCase):

    def test_cfg_filter_activation_only_nodes(self):
        input_shape = (8, 8, 3)
        in_model = create_model_conv2d_relu(input_shape)

        graph = prepare_graph(in_model,
                              base_config=get_base_config(),
                              bitwidth_candidates=get_full_bitwidth_candidates())

        # Filtering nodes; candidates
        filtered_graph = filter_nodes_candidates(graph)

        filtered_configurable_nodes = filtered_graph.get_configurable_sorted_nodes()

        # checking that layers with activation only (input and relu) have filtered configurations list,
        # that they have a configuration for each of the original bitwidth options
        input_candidates = filtered_configurable_nodes[0].candidates_quantization_cfg
        self.assertTrue(len(input_candidates) == 3,
                        f"Expects 3 input layer candidates, number of candidates is {len(input_candidates)}")
        self.assertTrue([c.activation_quantization_cfg.activation_n_bits for c in input_candidates] == [8, 4, 2])

        relu_candidates = filtered_configurable_nodes[2].candidates_quantization_cfg
        self.assertTrue(len(relu_candidates) == 3,
                        f"Expects 3 input layer candidates, number of candidates is {len(relu_candidates)}")
        self.assertTrue([c.activation_quantization_cfg.activation_n_bits for c in relu_candidates] == [8, 4, 2])

    def test_cfg_filter_weights_disabled(self):
        input_shape = (8, 8, 3)
        in_model = create_model_conv2d_only(input_shape)

        graph = prepare_graph(in_model,
                              base_config=get_base_config().clone_and_edit(enable_weights_quantization=False),
                              bitwidth_candidates=get_full_bitwidth_candidates())

        # Filtering nodes; candidates
        filtered_graph = filter_nodes_candidates(graph)

        filtered_configurable_nodes = filtered_graph.get_configurable_sorted_nodes()

        # checking that layers with weights (conv2d) have filtered activation configurations list
        # when weights quantization is disabled
        conv2d_1_candidates = filtered_configurable_nodes[1].candidates_quantization_cfg
        self.assertTrue(len(conv2d_1_candidates) == 3,
                        f"Expects 3 Conv layer candidates, number of candidates is {len(conv2d_1_candidates)}")
        self.assertTrue([c.activation_quantization_cfg.activation_n_bits for c in conv2d_1_candidates] == [8, 4, 2])
        conv2d_2_candidates = filtered_configurable_nodes[1].candidates_quantization_cfg
        self.assertTrue(len(conv2d_2_candidates) == 3,
                        f"Expects 3 Conv layer candidates, number of candidates is {len(conv2d_2_candidates)}")
        self.assertTrue([c.activation_quantization_cfg.activation_n_bits for c in conv2d_2_candidates] == [8, 4, 2])

    def test_cfg_filter_activation_disabled(self):
        input_shape = (8, 8, 3)
        in_model = create_model_conv2d_relu(input_shape)

        graph = prepare_graph(in_model,
                              base_config=get_base_config().clone_and_edit(enable_activation_quantization=False),
                              bitwidth_candidates=get_full_bitwidth_candidates())

        # Filtering nodes; candidates
        filtered_graph = filter_nodes_candidates(graph)

        filtered_configurable_nodes = filtered_graph.get_configurable_sorted_nodes()

        # checking that layers with weights (conv2d) have filtered weights configurations list
        # when activation quantization is disabled
        conv2d_candidates = filtered_configurable_nodes[0].candidates_quantization_cfg
        self.assertTrue(len(conv2d_candidates) == 3,
                        f"Expects 3 Conv layer candidates, number of candidates is {len(conv2d_candidates)}")
        self.assertTrue([c.weights_quantization_cfg.weights_n_bits for c in conv2d_candidates] == [8, 4, 2])

    def test_cfg_filter_multiple_candidates_weights_disabled(self):
        input_shape = (8, 8, 3)
        in_model = create_model_single_conv2d(input_shape)

        graph = prepare_graph(in_model,
                              base_config=get_base_config().clone_and_edit(enable_weights_quantization=False),
                              bitwidth_candidates=[(8, 8), (4, 8), (2, 8)])

        # Filtering nodes; candidates
        filtered_graph = filter_nodes_candidates(graph)

        filtered_graph_nodes = list(filtered_graph.nodes)

        # checking that layers with weights (conv2d) have filtered weights configurations list
        # when activation quantization is disabled
        conv2d_candidates = filtered_graph_nodes[1].candidates_quantization_cfg
        self.assertTrue(len(conv2d_candidates) == 1,
                        f"Expects 1 Conv layer candidates, number of candidates is {len(conv2d_candidates)}")
        candidate = conv2d_candidates[0]
        self.assertTrue((candidate.weights_quantization_cfg.weights_n_bits,
                         candidate.activation_quantization_cfg.activation_n_bits) == (DEFAULT_CANDIDATE_BITWIDTH, 8))


if __name__ == '__main__':
    unittest.main()
