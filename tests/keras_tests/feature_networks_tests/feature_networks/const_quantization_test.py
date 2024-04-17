# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
import numpy as np

import model_compression_toolkit as mct
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.common_tests.helpers.tensors_compare import cosine_similarity
from mct_quantizers import KerasQuantizationWrapper

from model_compression_toolkit.constants import TENSORFLOW
from model_compression_toolkit.target_platform_capabilities.constants import IMX500_TP_MODEL

keras = tf.keras
layers = keras.layers
tp = mct.target_platform


class ConstQuantizationTest(BaseKerasFeatureNetworkTest):

    def __init__(self, unit_test, layer, const, is_list_input=False, input_reverse_order=False, use_kwrags=False,
                 input_shape=(32, 32, 16)):
        super(ConstQuantizationTest, self).__init__(unit_test=unit_test, input_shape=input_shape)
        self.layer = layer
        self.const = const
        self.is_list_input = is_list_input
        self.input_reverse_order = input_reverse_order
        self.use_kwrags = use_kwrags

    def generate_inputs(self):
        # need positive inputs so won't divide with zero or take root of negative number
        return [1 + np.random.random(in_shape) for in_shape in self.get_input_shapes()]

    def get_tpc(self):
        return mct.get_target_platform_capabilities(TENSORFLOW, IMX500_TP_MODEL, "v2")

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = inputs
        if self.is_list_input:
            if self.input_reverse_order:
                x = self.layer([self.const, x])
            else:
                x = self.layer([x, self.const])
        else:
            if self.input_reverse_order:
                if self.use_kwrags:
                    x = self.layer(x=self.const, y=x)
                else:
                    x = self.layer(self.const, x)
            else:
                if self.use_kwrags:
                    x = self.layer(x=x, y=self.const)
                else:
                    x = self.layer(x, self.const)
        return tf.keras.models.Model(inputs=inputs, outputs=x)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'out shape is not as expected!')
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1, atol=0.001), msg=f'fail cosine similarity check:{cs}')
        self.unit_test.assertTrue(isinstance(quantized_model.layers[2], KerasQuantizationWrapper),
                                  msg='TFOpLambda should be quantized')
        const_index = 0 if self.input_reverse_order else 1
        self.unit_test.assertTrue((quantized_model.layers[2].weight_values[const_index] == self.const).all(),
                                  msg='Constant value should not change')
