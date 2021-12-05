# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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


from model_compression_toolkit.keras.default_framework_info import DEFAULT_KERAS_INFO
import unittest
import numpy as np
import model_compression_toolkit as mct
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
        activation_threshold_selection = mct.ThresholdSelectionMethod.NOCLIPPING
        for bias_correction in [True, False]:
            for weights_threshold_selection in [mct.ThresholdSelectionMethod.MSE,
                                                mct.ThresholdSelectionMethod.NOCLIPPING,
                                                mct.ThresholdSelectionMethod.MAE,
                                                mct.ThresholdSelectionMethod.LP,
                                                mct.ThresholdSelectionMethod.KL]:
                for weights_per_channel_threshold in [False, True]:
                    qc = mct.QuantizationConfig(activation_threshold_selection,
                                                weights_threshold_selection,
                                                mct.QuantizationMethod.POWER_OF_TWO,
                                                mct.QuantizationMethod.POWER_OF_TWO,
                                                activation_n_bits=8,
                                                weights_n_bits=16,
                                                relu_unbound_correction=relu_unbound_correction,
                                                weights_bias_correction=bias_correction,
                                                weights_per_channel_threshold=weights_per_channel_threshold)
                    q_model, quantization_info = mct.keras_post_training_quantization(model,
                                                                                      representative_data_gen,
                                                                                      n_iter=1,
                                                                                      quant_config=qc,
                                                                                      fw_info=DEFAULT_KERAS_INFO)

        for relu_unbound_correction in [True, False]:
            for activation_threshold_selection in [mct.ThresholdSelectionMethod.MSE,
                                                   mct.ThresholdSelectionMethod.NOCLIPPING,
                                                   mct.ThresholdSelectionMethod.MAE,
                                                   mct.ThresholdSelectionMethod.LP,
                                                   mct.ThresholdSelectionMethod.KL]:
                qc = mct.QuantizationConfig(activation_threshold_selection,
                                            weights_threshold_selection,
                                            mct.QuantizationMethod.POWER_OF_TWO,
                                            mct.QuantizationMethod.POWER_OF_TWO,
                                            activation_n_bits=8,
                                            weights_n_bits=16,
                                            relu_unbound_correction=relu_unbound_correction,
                                            weights_bias_correction=bias_correction,
                                            weights_per_channel_threshold=weights_per_channel_threshold)
                q_model, quantization_info = mct.keras_post_training_quantization(model,
                                                                                  representative_data_gen,
                                                                                  n_iter=1,
                                                                                  quant_config=qc,
                                                                                  fw_info=DEFAULT_KERAS_INFO)


if __name__ == '__main__':
    unittest.main()
