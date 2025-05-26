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


from model_compression_toolkit.core import MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.common.mixed_precision.sensitivity_eval.sensitivity_evaluation import SensitivityEvaluation
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2keras import \
    AttachTpcToKeras
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


def nms_output_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(1, 3, padding='same')(inputs)
    x = layers.ReLU()(x)

    # Dummy layers for creating NMS inputs with the required shape
    x = tf.squeeze(x, -1)
    x = tf.concat([x, x], -1)
    y = tf.concat([x, x], -1)
    y = tf.concat([y, y], -1)
    scores = tf.concat([x, y], -1) # shape = (batch, detections, classes)
    boxes, _ = tf.split(x, (4, 12), -1)
    boxes = tf.expand_dims(boxes, 2) # shape = (batch, detections, 1, box coordinates)

    # NMS layer
    outputs = tf.image.combined_non_max_suppression(
        boxes,
        scores,
        max_output_size_per_class=300,
        max_total_size=300,
        iou_threshold=0.7,
        score_threshold=0.001,
        pad_per_class=False,
        clip_boxes=False
    )
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def representative_dataset():
    for _ in range(2):
        yield [np.random.randn(1, 8, 8, 3).astype(np.float32)]


class TestSensitivityEvalWithNonSupportedOutputNodes(unittest.TestCase):

    def verify_test_for_model(self, model):
        keras_impl = KerasImplementation()
        graph = prepare_graph_with_quantization_parameters(model,
                                                           keras_impl,
                                                           DEFAULT_KERAS_INFO,
                                                           representative_dataset,
                                                           generate_keras_tpc,
                                                           attach2fw=AttachTpcToKeras(),
                                                           input_shape=(1, 8, 8, 3),
                                                           mixed_precision_enabled=True)

        hessian_info_service = hess.HessianInfoService(graph=graph,
                                                       fw_impl=keras_impl)

        # Reducing the default number of samples for Mixed precision Hessian approximation
        # to allow quick execution of the test
        se = SensitivityEvaluation(graph, MixedPrecisionQuantizationConfig(use_hessian_based_scores=True,
                                                                           num_of_images=2), representative_dataset,
                                   DEFAULT_KERAS_INFO, keras_impl, hessian_info_service=hessian_info_service)

    def test_not_supported_output_argmax(self):
        model = argmax_output_model((8, 8, 3))
        with self.assertRaises(Exception) as e:
            self.verify_test_for_model(model)
        self.assertTrue("All graph outputs must support Hessian score computation" in str(e.exception))

    def test_not_supported_output_nms(self):
        model = nms_output_model((8, 8, 3))
        with self.assertRaises(Exception) as e:
            self.verify_test_for_model(model)
        self.assertTrue("All graph outputs must support Hessian score computation" in str(e.exception))


if __name__ == '__main__':
    unittest.main()
