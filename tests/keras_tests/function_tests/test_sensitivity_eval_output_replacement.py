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
import numpy as np
import tensorflow as tf

from keras.applications.densenet import DenseNet121
from keras.applications.mobilenet_v2 import MobileNetV2

from model_compression_toolkit.core import MixedPrecisionQuantizationConfigV2
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_configs, \
    prepare_graph_with_quantization_parameters
import model_compression_toolkit.core.common.hessian as hess

keras = tf.keras
layers = keras.layers


def argmax_output_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 4)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 4)(x)
    x = layers.ReLU()(x)
    outputs = tf.argmax(x, axis=-1)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def softmax_output_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 4)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 4)(x)
    x = layers.ReLU()(x)
    outputs = tf.nn.softmax(x, axis=-1)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def representative_dataset():
    yield [np.random.randn(1, 16, 16, 3).astype(np.float32)]


class TestSensitivityEvalWithOutputReplacementNodes(unittest.TestCase):

    def verify_test_for_model(self, model):
        keras_impl = KerasImplementation()
        graph = prepare_graph_with_quantization_parameters(model,
                                                           keras_impl,
                                                           DEFAULT_KERAS_INFO,
                                                           representative_dataset,
                                                           generate_keras_tpc,
                                                           input_shape=(1, 16, 16, 3),
                                                           mixed_precision_enabled=True)

        hess.hessian_service.set_graph(graph=graph)
        hess.hessian_service.set_fw_impl(keras_impl)

        se = keras_impl.get_sensitivity_evaluator(graph,
                                                  MixedPrecisionQuantizationConfigV2(use_grad_based_weights=True),
                                                  representative_dataset,
                                                  DEFAULT_KERAS_INFO)

        # If the output replacement nodes for MP sensitivity evaluation has been computed correctly then the ReLU layer
        # should be added to the interest points and included in the output nodes list for metric computation purposes.
        relu_node = graph.get_topo_sorted_nodes()[-2]
        self.assertTrue(relu_node.type == layers.ReLU)
        self.assertIn(relu_node, se.interest_points)
        self.assertEqual(len(se.outputs_replacement_nodes), 1)
        self.assertIn(relu_node, se.outputs_replacement_nodes)
        self.assertEqual(se.output_nodes_indices, [2, 3])

    def test_output_replacement_argmax(self):
        model = argmax_output_model((16, 16, 3))
        self.verify_test_for_model(model)

    def test_output_replacement_softmax(self):
        model = softmax_output_model((16, 16, 3))
        self.verify_test_for_model(model)


if __name__ == '__main__':
    unittest.main()
