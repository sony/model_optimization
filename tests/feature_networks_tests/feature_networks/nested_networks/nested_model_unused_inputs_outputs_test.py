# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


from tensorflow.python.keras.engine.functional import Functional
from tensorflow.python.keras.engine.sequential import Sequential
from tests.feature_networks_tests.base_feature_test import BaseFeatureNetworkTest
import network_optimization_package as snop
import tensorflow as tf
import numpy as np
from tests.helpers.tensors_compare import cosine_similarity


keras = tf.keras
layers = keras.layers


class NestedModelUnusedInputsOutputsTest(BaseFeatureNetworkTest):
    """
    Test two inner models when one's some outputs are not connected to any layer.
    """
    def __init__(self, unit_test):
        super().__init__(unit_test, num_calibration_iter=1, val_batch_size=32)

    def get_quantization_config(self):
        return snop.QuantizationConfig(snop.ThresholdSelectionMethod.MSE, snop.ThresholdSelectionMethod.MSE,
                                      snop.QuantizationMethod.SYMMETRIC_UNIFORM, snop.QuantizationMethod.SYMMETRIC_UNIFORM,
                                      16, 16, True, True, True)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 236, 236, 3]]

    def inner_functional_model(self, input_shape):
        inputs = layers.Input(shape=input_shape[1:])
        x = layers.Conv2D(3, 4)(inputs)
        y = layers.Conv2D(3, 4)(inputs)
        z = layers.Conv2D(3, 4)(inputs)
        w = layers.Conv2D(3, 4)(inputs)
        x = layers.BatchNormalization()(x)
        outputs = layers.Activation('swish')(x)
        return keras.Model(inputs=inputs, outputs=[outputs,y,z,w])

    def second_inner_functional_model(self, input_shape):
        inputs = layers.Input(shape=input_shape[1:])
        inputs2 = layers.Input(shape=input_shape[1:])
        x = layers.Conv2D(3, 4)(inputs)
        y = layers.Conv2D(3, 4)(inputs2)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x,y])
        outputs = layers.Activation('swish')(x)
        return keras.Model(inputs=[inputs,inputs2], outputs=outputs)


    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.Conv2D(3,4)(inputs)
        x = layers.Conv2D(3,4)(x)
        x,y,z,w = self.inner_functional_model(x.shape)(x)
        x = self.second_inner_functional_model(x.shape)([z,y])
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        for l in quantized_model.layers:
            if hasattr(l, 'layer'):
                self.unit_test.assertFalse(isinstance(l.layer, Functional) or isinstance(l.layer, Sequential))
            else:
                self.unit_test.assertFalse(isinstance(l, Functional) or isinstance(l, Sequential))
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')
