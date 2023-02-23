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

import tensorflow as tf

from model_compression_toolkit.quantizers_infrastructure import KerasQuantizationWrapper
from tests.quantizers_infrastructure_tests.trainable_infrastructure_tests.keras.base_keras_trainable_infra_test import \
    BaseKerasTrainableInfrastructureTest, \
    IdentityWeightsQuantizer, ZeroActivationsQuantizer

keras = tf.keras
layers = keras.layers

WEIGHT = 'kernel'
CLASS_NAME = 'class_name'


class TestKerasWeightsQuantizationWrapper(BaseKerasTrainableInfrastructureTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(6, 7, use_bias=False)(inputs)
        return keras.Model(inputs=inputs, outputs=x)

    def run_test(self):
        model = self.create_networks()
        conv_layer = model.layers[1]
        inputs = self.generate_inputs()[0]

        wrapper = self.get_wrapper(conv_layer)
        wrapper.add_weights_quantizer(WEIGHT, IdentityWeightsQuantizer(self.get_weights_quantization_config()))

        # build
        wrapper.build(self.get_input_shapes())
        (name, weight, quantizer) = wrapper._weights_vars[0]
        self.unit_test.assertTrue(isinstance(wrapper, KerasQuantizationWrapper))
        self.unit_test.assertTrue(isinstance(wrapper.layer, layers.Conv2D))
        self.unit_test.assertTrue(name == WEIGHT)
        self.unit_test.assertTrue((weight == getattr(wrapper.layer, WEIGHT)).numpy().all())
        self.unit_test.assertTrue(isinstance(quantizer, IdentityWeightsQuantizer))

        # call
        outputs = wrapper.call(inputs.astype('float32'))
        self.unit_test.assertTrue((outputs == conv_layer(inputs)).numpy().all())


class TestKerasActivationsQuantizationWrapper(TestKerasWeightsQuantizationWrapper):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def run_test(self):
        model = self.create_networks()
        conv_layer = model.layers[1]
        inputs = self.generate_inputs()[0]

        wrapper = self.get_wrapper(conv_layer, activation_quantizers=[ZeroActivationsQuantizer(
            self.get_activation_quantization_config())])

        # build
        wrapper.build(self.get_input_shapes())
        (act_quantizer) = wrapper._activation_vars[0]
        self.unit_test.assertTrue(isinstance(act_quantizer, ZeroActivationsQuantizer))

        # apply the wrapper on inputs
        outputs = wrapper.call(inputs.astype('float32'))
        self.unit_test.assertTrue((outputs == 0).numpy().all())
