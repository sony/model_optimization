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
import unittest
import keras
import numpy as np

from keras import Input
from keras.layers import Conv2D
from mct_quantizers import KerasQuantizationWrapper, KerasActivationQuantizationHolder

from model_compression_toolkit.core.common.mixed_precision.set_layer_to_bitwidth import set_layer_to_bitwidth
from model_compression_toolkit.core.keras.constants import KERNEL
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.core.keras.mixed_precision.configurable_activation_quantizer import \
    ConfigurableActivationQuantizer
from model_compression_toolkit.core.keras.mixed_precision.configurable_weights_quantizer import \
    ConfigurableWeightsQuantizer
from model_compression_toolkit.target_platform_capabilities.tpc_models.default_tpc.latest import generate_keras_tpc
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_quantization_parameters


def base_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(2, 3)(inputs)
    return keras.Model(inputs=inputs, outputs=x)


def representative_dataset():
    yield [np.random.randn(1, 8, 8, 3).astype(np.float32)]


def test_setup():

    model = base_model((8, 8, 3))

    graph = prepare_graph_with_quantization_parameters(model,  KerasImplementation(), DEFAULT_KERAS_INFO,
                                                       representative_dataset, generate_keras_tpc, input_shape=(1, 8, 8, 3))

    layer = model.layers[1]
    node = graph.get_topo_sorted_nodes()[1]

    return layer, node


class TestKerasSetLayerToBitwidth(unittest.TestCase):

    def test_set_layer_to_bitwidth_weights(self):
        layer, node = test_setup()

        wrapper_layer = \
            KerasQuantizationWrapper(layer,
                                     weights_quantizers={KERNEL:
                                         ConfigurableWeightsQuantizer(
                                             node_q_cfg=node.candidates_quantization_cfg,
                                             float_weights=node.get_weights_by_keys(KERNEL),
                                             max_candidate_idx=node.find_max_candidates_indices()[0])
                                     })

        for attr, q in wrapper_layer.weights_quantizers.items():
            self.assertEqual(q.active_quantization_config_index, 0)
            # Changing active quantizer candidate index manually to 1 (this is an invalid value in this case)
            q.active_quantization_config_index = 1

        set_layer_to_bitwidth(wrapper_layer, bitwidth_idx=0, weights_quantizer_type=ConfigurableWeightsQuantizer,
                              activation_quantizer_type=ConfigurableActivationQuantizer,
                              weights_quant_layer_type=KerasQuantizationWrapper,
                              activation_quant_layer_type=KerasActivationQuantizationHolder)

        for attr, q in wrapper_layer.weights_quantizers.items():
            self.assertEqual(q.active_quantization_config_index, 0)

    def test_set_layer_to_bitwidth_activation(self):
        layer, node = test_setup()

        holder_layer = \
            KerasActivationQuantizationHolder(ConfigurableActivationQuantizer(
                node_q_cfg=node.candidates_quantization_cfg,
                max_candidate_idx=node.find_max_candidates_indices()[0]))

        q = holder_layer.activation_holder_quantizer

        self.assertEqual(q.active_quantization_config_index, 0)

        # Changing active quantizer candidate index manually to 1 (this is an invalid value in this case)
        q.active_quantization_config_index = 1

        set_layer_to_bitwidth(holder_layer, bitwidth_idx=0, weights_quantizer_type=ConfigurableWeightsQuantizer,
                              activation_quantizer_type=ConfigurableActivationQuantizer,
                              weights_quant_layer_type=KerasQuantizationWrapper,
                              activation_quant_layer_type=KerasActivationQuantizationHolder)

        self.assertEqual(q.active_quantization_config_index, 0)
