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


from tests.feature_networks_tests.base_feature_test import BaseFeatureNetworkTest
import network_optimization_package as snop
import tensorflow as tf
import numpy as np
from tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class ShiftNegActivationTest(BaseFeatureNetworkTest):
    def __init__(self, unit_test, linear_op_to_test, use_pad_layer=False):
        assert type(linear_op_to_test) in [layers.Conv2D, layers.Dense, layers.DepthwiseConv2D]
        self.linear_op_to_test = linear_op_to_test
        self.use_pad_layer = use_pad_layer
        super().__init__(unit_test, num_calibration_iter=1, val_batch_size=32)

    def get_quantization_config(self):
        return snop.QuantizationConfig(snop.ThresholdSelectionMethod.MSE,
                                       snop.ThresholdSelectionMethod.MSE,
                                       snop.QuantizationMethod.SYMMETRIC_UNIFORM,
                                       snop.QuantizationMethod.SYMMETRIC_UNIFORM,
                                       16,
                                       16,
                                       False,
                                       False,
                                       True,
                                       shift_negative_activation_correction=True)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 224, 244, 3]]

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.Activation('swish')(inputs)
        if self.use_pad_layer:
            x = layers.ZeroPadding2D(((3, 4), (5, 6)))(x)
        outputs = self.linear_op_to_test(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        w, b = float_model.get_weights()
        linear_op_index = 3 + (4 if self.use_pad_layer else 3)
        linear_op_index = linear_op_index + int(self.linear_op_to_test.get_config().get('padding') == 'same')
        q_w, q_b = quantized_model.layers[linear_op_index].weights[0].numpy(), \
                   quantized_model.layers[linear_op_index].weights[1].numpy()
        # Take the ACTUAL value the activations were shifted by, from the Add layer the substitution needs to insert
        add_layer_index = 4  # should be always the fourth layer (input, fq, swish, add)
        shift_nl_out = quantized_model.layers[add_layer_index].constants[1].item()
        if isinstance(self.linear_op_to_test, layers.DepthwiseConv2D):
            self.unit_test.assertTrue(np.allclose(b - q_b, shift_nl_out * np.sum(w, axis=(0, 1)).flatten()))
        elif isinstance(self.linear_op_to_test, layers.Conv2D):
            self.unit_test.assertTrue(np.allclose(b - q_b, shift_nl_out * np.sum(w, axis=(0, 1, 2))))
        elif isinstance(self.linear_op_to_test, layers.Dense):
            self.unit_test.assertTrue(np.allclose(b - q_b, shift_nl_out * np.sum(w, axis=(0))))
        else:
            raise NotImplementedError

        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')
