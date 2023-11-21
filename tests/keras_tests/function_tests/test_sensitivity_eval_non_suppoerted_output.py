# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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


from model_compression_toolkit.core import MixedPrecisionQuantizationConfigV2
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_quantization_parameters
import model_compression_toolkit.core.common.hessian as hess

keras = tf.keras
layers = keras.layers


def argmax_output_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(3, 3)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(3, 3)(x)
    x = layers.ReLU()(x)
    outputs = tf.argmax(x, axis=-1)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def representative_dataset():
    yield [np.random.randn(1, 8, 8, 3).astype(np.float32)]


class TestSensitivityEvalWithNonSupportedOutputNodes(unittest.TestCase):

    def verify_test_for_model(self, model):
        keras_impl = KerasImplementation()
        graph = prepare_graph_with_quantization_parameters(model,
                                                           keras_impl,
                                                           DEFAULT_KERAS_INFO,
                                                           representative_dataset,
                                                           generate_keras_tpc,
                                                           input_shape=(1, 8, 8, 3),
                                                           mixed_precision_enabled=True)

        hessian_info_service = hess.HessianInfoService(graph=graph,
                                                       representative_dataset=representative_dataset,
                                                       fw_impl=keras_impl)

        se = keras_impl.get_sensitivity_evaluator(graph,
                                                  MixedPrecisionQuantizationConfigV2(use_hessian_based_scores=True),
                                                  representative_dataset,
                                                  DEFAULT_KERAS_INFO,
                                                  hessian_info_service=hessian_info_service)

    def test_not_supported_output_argmax(self):
        model = argmax_output_model((8, 8, 3))
        with self.assertRaises(Exception) as e:
            self.verify_test_for_model(model)
        self.assertTrue("All graph outputs should support Hessian computation" in str(e.exception))


if __name__ == '__main__':
    unittest.main()
