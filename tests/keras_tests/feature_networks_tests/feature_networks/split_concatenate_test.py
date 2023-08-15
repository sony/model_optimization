# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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


import tensorflow as tf
from packaging import version
if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers.core import TFOpLambda
else:
    from keras.layers.core import TFOpLambda

from mct_quantizers import KerasActivationQuantizationHolder
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers


class SplitConcatenateTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, experimental_exporter=True)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Dense(30)(inputs)
        output_split = tf.split(x, num_or_size_splits=30, axis=-1)
        y = layers.Conv2D(3, 4)(output_split[1])
        z = layers.Conv2D(3, 4)(output_split[3])
        outputs = layers.Concatenate()([y, z])
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        split_layer = get_layers_from_model_by_type(quantized_model, TFOpLambda)[0]
        self.unit_test.assertTrue(split_layer.function == tf.split)
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)

        self.unit_test.assertTrue(split_layer.output[1].ref() == conv_layers[0].input.ref())
        self.unit_test.assertTrue(split_layer.output[3].ref() == conv_layers[1].input.ref())

        concate_layer = get_layers_from_model_by_type(quantized_model, layers.Concatenate)[0]
        self.unit_test.assertTrue(len(concate_layer.input) == 2)

        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        self.unit_test.assertTrue(holder_layers[-3].input.ref() == conv_layers[0].output.ref())
        self.unit_test.assertTrue(holder_layers[-2].input.ref() == conv_layers[1].output.ref())
        self.unit_test.assertTrue(holder_layers[-3].output.ref() == concate_layer.input[0].ref())
        self.unit_test.assertTrue(holder_layers[-2].output.ref() == concate_layer.input[1].ref())

