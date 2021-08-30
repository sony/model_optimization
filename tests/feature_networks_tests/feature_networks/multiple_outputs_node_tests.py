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
import sony_model_optimization_package as smop
import tensorflow as tf
import numpy as np
from tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class MultipleOutputsNodeTests(BaseFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, num_calibration_iter=1, val_batch_size=32)

    def get_quantization_config(self):
        return smop.QuantizationConfig(smop.ThresholdSelectionMethod.NOCLIPPING, smop.ThresholdSelectionMethod.NOCLIPPING,
                                       smop.QuantizationMethod.SYMMETRIC_UNIFORM, smop.QuantizationMethod.SYMMETRIC_UNIFORM,
                                       16, 16, True, False, True)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 224, 244, 3]]

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
        x = layers.Dense(20)(inputs)
        x = layers.ReLU(max_value=6.0)(x)
        outputs = layers.Dense(20)(x)
        outputs = tf.split(outputs, num_or_size_splits=20, axis=-1)
        return keras.Model(inputs=inputs, outputs=[outputs[0], outputs[4], outputs[2]])

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(len(quantized_model.outputs) == 3)
        inputs = self.generate_inputs(self.create_inputs_shape())
        output_q = quantized_model.predict(inputs)
        output_f = float_model.predict(inputs)
        for o_q, o_f in zip(output_q, output_f):
            cs = cosine_similarity(o_f, o_q)
            self.unit_test.assertTrue(np.isclose(cs, 1))
