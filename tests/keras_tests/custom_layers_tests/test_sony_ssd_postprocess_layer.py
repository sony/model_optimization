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

import model_compression_toolkit as mct
from sony_custom_layers.keras.object_detection.ssd_post_process import SSDPostProcess
from mct_quantizers.keras.metadata import MetadataLayer

keras = tf.keras
layers = keras.layers


def get_rep_dataset(n_iters, in_shape):
    def rep_dataset():
        for _ in range(n_iters):
            yield [np.random.randn(*in_shape)]

    return rep_dataset


class TestSonySsdPostProcessLayer(unittest.TestCase):

    def test_custom_layer(self):
        inputs = layers.Input(shape=(8, 8, 3))
        x = layers.Conv2D(32, 4)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, 4)(x)
        x = layers.ReLU()(x)
        x = layers.Reshape((32, 4))(x)
        ssd_pp = SSDPostProcess(tf.constant(np.random.random(size=list(x.shape[1:])), dtype=tf.float32), [1, 1, 1, 1],
                                [8, 8], 'sigmoid', score_threshold=0.001,
                                iou_threshold=0.5, max_detections=10)
        outputs = ssd_pp((x, x))
        model = keras.Model(inputs=inputs, outputs=outputs)

        core_config = mct.core.CoreConfig(
            mixed_precision_config=mct.core.MixedPrecisionQuantizationConfig(
                use_hessian_based_scores=False))
        q_model, _ = mct.ptq.keras_post_training_quantization(model,
                                                              get_rep_dataset(2, (1, 8, 8, 3)),
                                                              core_config=core_config,
                                                              target_resource_utilization=mct.core.ResourceUtilization(weights_memory=6000))

        # verify the custom layer is in the quantized model
        last_model_layer_index = -2 if isinstance(q_model.layers[-1], MetadataLayer) else -1
        self.assertTrue(isinstance(q_model.layers[last_model_layer_index], SSDPostProcess), 'Custom layer should be in the quantized model')
        # verify mixed-precision
        self.assertTrue(any([q_model.layers[2].weights_quantizers['kernel'].num_bits < 8,
                             q_model.layers[4].weights_quantizers['kernel'].num_bits < 8]))
