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


from tests.feature_networks_tests.base_feature_test import BaseFeatureNetworkTest
import model_compression_toolkit as mct
from model_compression_toolkit.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.common.user_info import UserInformation
import tensorflow as tf
import numpy as np
from tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class GradientPTQBaseTest(BaseFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.ThresholdSelectionMethod.NOCLIPPING, mct.ThresholdSelectionMethod.NOCLIPPING,
                                      mct.QuantizationMethod.POWER_OF_TWO, mct.QuantizationMethod.POWER_OF_TWO,
                                      16, 16, True, False, True)


    def get_gptq_config(self):
        return mct.GradientPTQConfig(5)

    def create_feature_network(self, input_shape):
        inputs = layers.Input(shape=input_shape[0][1:])
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
        input_shapes = self.create_inputs_shape()
        x = self.generate_inputs(input_shapes)

        def representative_data_gen():
            return x

        model_float = self.create_feature_network(input_shapes)

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
        return mct.GradientPTQConfig(50,
                                     optimizer=tf.keras.optimizers.SGD(learning_rate=50.0))

    def compare(self, quantized_model, quantized_gptq_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(len(quantized_model.weights) == len(quantized_gptq_model.weights),
                                  msg='float model number of weights different from quantized model: ' +
                                      f'{len(quantized_gptq_model.weights)} != {len(quantized_model.weights)}')
        # check all weights were updated
        weights_mean_diff = [np.mean(w_q.numpy() != w_f.numpy()) > 0.3 for w_q, w_f in zip(quantized_model.weights,
                                                                                           quantized_gptq_model.weights)]
        self.unit_test.assertTrue(all(weights_mean_diff), msg="Some weights weren't updated")


class GradientPTQLearnRateZeroTest(GradientPTQBaseTest):

    def get_gptq_config(self):
        return mct.GradientPTQConfig(1,
                                     optimizer=tf.keras.optimizers.SGD(learning_rate=0.0))

    def compare(self, quantized_model, quantized_gptq_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(len(quantized_model.weights) == len(quantized_gptq_model.weights),
                                  msg='float model number of weights different from quantized model: ' +
                                      f'{len(quantized_gptq_model.weights)} != {len(quantized_model.weights)}')
        # check all weights didn't change
        weights_diff = [np.all(w_q.numpy() == w_f.numpy()) for w_q, w_f in
                        zip(quantized_model.weights, quantized_gptq_model.weights)]
        self.unit_test.assertTrue(all(weights_diff), msg="Some weights were updated")
