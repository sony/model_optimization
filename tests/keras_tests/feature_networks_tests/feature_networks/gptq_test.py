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
import numpy as np
import tensorflow as tf

import model_compression_toolkit as mct
import model_compression_toolkit.common.gptq.gptq_config
from model_compression_toolkit.common.user_info import UserInformation
from model_compression_toolkit.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.keras.gradient_ptq.gptq_loss import multiple_tensors_mse_loss
from tests.common_tests.helpers.tensors_compare import cosine_similarity
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

keras = tf.keras
layers = keras.layers


class GradientPTQBaseTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test,
                         input_shape=(1,16,16,3))

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.QuantizationErrorMethod.NOCLIPPING, mct.QuantizationErrorMethod.NOCLIPPING,16, 16,
                                      True, False, True)


    def get_gptq_config(self):
        return model_compression_toolkit.common.gptq.gptq_config.GradientPTQConfig(5,
                                                                                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                                                                                   loss=multiple_tensors_mse_loss)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(3, 4)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(7, 8)(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def compare(self, ptq_model, model_float, input_x=None, quantization_info: UserInformation = None):
        raise NotImplementedError(f'{self.__class__} did not implement compare')

    def run_test(self):

        x = self.generate_inputs()

        def representative_data_gen():
            return x

        model_float = self.create_networks()

        qc = self.get_quantization_config()
        ptq_model, quantization_info = mct.keras_post_training_quantization(model_float, representative_data_gen,
                                                                            n_iter=self.num_calibration_iter,
                                                                            quant_config=qc,
                                                                            fw_info=DEFAULT_KERAS_INFO,
                                                                            network_editor=self.get_network_editor())
        ptq_gptq_model, quantization_info = mct.keras_post_training_quantization(model_float, representative_data_gen,
                                                                                 n_iter=self.num_calibration_iter,
                                                                                 quant_config=qc,
                                                                                 fw_info=DEFAULT_KERAS_INFO,
                                                                                 network_editor=self.get_network_editor(),
                                                                                 gptq_config=self.get_gptq_config())

        self.compare(ptq_model, ptq_gptq_model, input_x=x, quantization_info=quantization_info)


class GradientPTQTest(GradientPTQBaseTest):

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check: {cs}')


class GradientPTQWeightsUpdateTest(GradientPTQBaseTest):

    def get_gptq_config(self):
        return model_compression_toolkit.common.gptq.gptq_config.GradientPTQConfig(50,
                                                                                   optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
                                                                                   loss=multiple_tensors_mse_loss)

    def compare(self, quantized_model, quantized_gptq_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(len(quantized_model.weights) == len(quantized_gptq_model.weights),
                                  msg='float model number of weights different from quantized model: ' +
                                      f'{len(quantized_gptq_model.weights)} != {len(quantized_model.weights)}')
        # check all weights were updated
        weights_diff = [np.any(w_q.numpy() != w_f.numpy()) for w_q, w_f in zip(quantized_model.weights,
                                                                               quantized_gptq_model.weights)]
        self.unit_test.assertTrue(all(weights_diff), msg="Some weights weren't updated")


class GradientPTQLearnRateZeroTest(GradientPTQBaseTest):

    def get_gptq_config(self):
        return model_compression_toolkit.common.gptq.gptq_config.GradientPTQConfig(1,
                                                                                   optimizer=tf.keras.optimizers.SGD(learning_rate=0.0),
                                                                                   loss=multiple_tensors_mse_loss)

    def compare(self, quantized_model, quantized_gptq_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(len(quantized_model.weights) == len(quantized_gptq_model.weights),
                                  msg='float model number of weights different from quantized model: ' +
                                      f'{len(quantized_gptq_model.weights)} != {len(quantized_model.weights)}')
        # check all weights didn't change (small noise is possible due to quantization with numpy
        # vs quantization with tf)
        weights_diff = [np.isclose(np.max(np.abs(w_q - w_f)), 0, atol=1e-5) for w_q, w_f in
                        zip(quantized_model.weights, quantized_gptq_model.weights)]
        for weights_close in weights_diff:
            self.unit_test.assertTrue(np.all(weights_close))
