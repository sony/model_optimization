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


from network_optimization_package.keras.default_framework_info import DEFAULT_KERAS_INFO
import unittest
import numpy as np
import network_optimization_package as snop
import tensorflow as tf
from tensorflow.keras import layers



def model_gen():
    inputs = layers.Input(shape=[16, 16, 3])
    x = layers.Conv2D(2, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=x)


class TestQuantizationConfigurations(unittest.TestCase):
    def test_run_quantization_config_mbv1(self):
        x = np.random.randn(1, 16, 16, 3)

        def representative_data_gen():
            return [x]

        model = model_gen()
        relu_unbound_correction = False
        activation_threshold_selection = snop.ThresholdSelectionMethod.NOCLIPPING
        for bias_correction in [True, False]:
            for weights_threshold_selection in [snop.ThresholdSelectionMethod.MSE,
                                                snop.ThresholdSelectionMethod.NOCLIPPING,
                                                snop.ThresholdSelectionMethod.MAE,
                                                snop.ThresholdSelectionMethod.LP,
                                                snop.ThresholdSelectionMethod.KL]:
                for weights_per_channel_threshold in [False, True]:
                    qc = snop.QuantizationConfig(activation_threshold_selection,
                                                weights_threshold_selection,
                                                snop.QuantizationMethod.SYMMETRIC_UNIFORM,
                                                snop.QuantizationMethod.SYMMETRIC_UNIFORM,
                                                activation_n_bits=8,
                                                weights_n_bits=16,
                                                relu_unbound_correction=relu_unbound_correction,
                                                weights_bias_correction=bias_correction,
                                                weights_per_channel_threshold=weights_per_channel_threshold)
                    q_model, quantization_info = snop.keras_post_training_quantization(model,
                                                                                      representative_data_gen,
                                                                                      n_iter=1,
                                                                                      quant_config=qc,
                                                                                      fw_info=DEFAULT_KERAS_INFO)

        for relu_unbound_correction in [True, False]:
            for activation_threshold_selection in [snop.ThresholdSelectionMethod.MSE,
                                                   snop.ThresholdSelectionMethod.NOCLIPPING,
                                                   snop.ThresholdSelectionMethod.MAE,
                                                   snop.ThresholdSelectionMethod.LP,
                                                   snop.ThresholdSelectionMethod.KL]:
                qc = snop.QuantizationConfig(activation_threshold_selection,
                                            weights_threshold_selection,
                                            snop.QuantizationMethod.SYMMETRIC_UNIFORM,
                                            snop.QuantizationMethod.SYMMETRIC_UNIFORM,
                                            activation_n_bits=8,
                                            weights_n_bits=16,
                                            relu_unbound_correction=relu_unbound_correction,
                                            weights_bias_correction=bias_correction,
                                            weights_per_channel_threshold=weights_per_channel_threshold)
                q_model, quantization_info = snop.keras_post_training_quantization(model,
                                                                                  representative_data_gen,
                                                                                  n_iter=1,
                                                                                  quant_config=qc,
                                                                                  fw_info=DEFAULT_KERAS_INFO)


if __name__ == '__main__':
    unittest.main()
