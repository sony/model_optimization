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

from model_compression_toolkit.quantizers_infrastructure import KerasNodeQuantizationDispatcher, \
    KerasQuantizationWrapper
from tests.keras_tests.infrastructure_tests.base_keras_infrastructure_test import BaseKerasInfrastructureTest, \
    IdentityQuantizer, weight_quantization_config, DISPATCHER, CLASS_NAME, ZeroActivationsQuantizer, \
    activation_quantization_config

keras = tf.keras
layers = keras.layers


class ConvLayerKerasInfrastructureTest(BaseKerasInfrastructureTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(6, 7, use_bias=False)(inputs)
        return keras.Model(inputs=inputs, outputs=x)

    def run_test(self, weight_str):
        # Check weights wrapper
        model = self.create_networks()
        conv_layer = model.layers[1]
        inputs = self.generate_inputs()[0]

        nqd = self.get_dispatcher()
        nqd.add_weight_quantizer(weight_str, IdentityQuantizer(weight_quantization_config))
        wrapper = self.get_wrapper(conv_layer, nqd)

        # get config
        wrapper_config = wrapper.get_config()
        self.unit_test.assertTrue(wrapper_config[DISPATCHER][CLASS_NAME] ==
                                  KerasNodeQuantizationDispatcher.__name__)

        # build
        wrapper.build(self.get_input_shapes())
        (name, weight, quantizer) = wrapper._weight_vars[0]
        self.unit_test.assertTrue(isinstance(wrapper, KerasQuantizationWrapper))
        self.unit_test.assertTrue(isinstance(wrapper.layer, layers.Conv2D))
        self.unit_test.assertTrue(name == weight_str)
        self.unit_test.assertTrue((weight == getattr(wrapper.layer, weight_str)).numpy().all())
        self.unit_test.assertTrue(isinstance(quantizer, IdentityQuantizer))

        # call
        outputs = wrapper.call(inputs.astype('float32'))
        self.unit_test.assertTrue((outputs == conv_layer(inputs)).numpy().all())

        # Check activations wrapper
        nqd = self.get_dispatcher(activation_quantizers=
                                  [ZeroActivationsQuantizer(activation_quantization_config)])
        wrapper = self.get_wrapper(conv_layer, nqd)
        # build
        wrapper.build(self.get_input_shapes())
        (act_quantizer) = wrapper._activation_vars[0]
        self.unit_test.assertTrue(isinstance(act_quantizer, ZeroActivationsQuantizer))
        outputs = wrapper.call(inputs.astype('float32'))
        self.unit_test.assertTrue((outputs == 0).numpy().all())

