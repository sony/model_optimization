# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import List

import numpy as np
import tensorflow as tf

import model_compression_toolkit as mct
import model_compression_toolkit.gptq.common.gptq_config
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.gptq.keras.gptq_loss import multiple_tensors_mse_loss
from tests.common_tests.helpers.tensors_compare import cosine_similarity
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.keras_tests.tpc_keras import get_16bit_tpc

keras = tf.keras
layers = keras.layers
tp = mct.target_platform


def build_model(in_input_shape: List[int]) -> keras.Model:
    """
    This function generate a simple network to test GPTQ
    Args:
        in_input_shape: Input shape list

    Returns:

    """
    inputs = layers.Input(shape=in_input_shape)
    x = layers.Conv2D(32, 4, bias_initializer='glorot_uniform')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.Conv2D(32, 8, bias_initializer='glorot_uniform')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.ReLU()(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


class GradientPTQBaseTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, is_gumbel=False, sam_optimization=False):
        super().__init__(unit_test,
                         input_shape=(1, 16, 16, 3))
        self.is_gumbel = is_gumbel
        self.sam_optimization = sam_optimization

    def get_tpc(self):
        return get_16bit_tpc("gptq_test")

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.QuantizationErrorMethod.NOCLIPPING, mct.QuantizationErrorMethod.NOCLIPPING,
                                      True, False, True)

    def get_gptq_config(self):
        return model_compression_toolkit.gptq.common.gptq_config.GradientPTQConfig(5,
                                                                                   optimizer=tf.keras.optimizers.Adam(
                                                                                       learning_rate=0.0001),
                                                                                   optimizer_rest=tf.keras.optimizers.Adam(
                                                                                       learning_rate=0.0001),
                                                                                   sam_optimization=self.sam_optimization,
                                                                                   loss=multiple_tensors_mse_loss,
                                                                                   rounding_type=model_compression_toolkit.gptq.common.gptq_config.RoundingType.GumbelRounding if self.is_gumbel else model_compression_toolkit.gptq.common.gptq_config.RoundingType.STE,
                                                                                   train_bias=True)

    def create_networks(self):
        in_shape = self.get_input_shapes()[0][1:]
        return build_model(in_shape)

    def compare(self, ptq_model, model_float, input_x=None, quantization_info: UserInformation = None):
        raise NotImplementedError(f'{self.__class__} did not implement compare')

    def run_test(self, experimental_facade=False, experimental_exporter=False):
        x = self.generate_inputs()

        def representative_data_gen():
            return x

        model_float = self.create_networks()

        qc = self.get_quantization_config()
        tpc = self.get_tpc()
        if experimental_facade:
            ptq_model, quantization_info = mct.keras_post_training_quantization_experimental(model_float,
                                                                              self.representative_data_gen,
                                                                              target_kpi=self.get_kpi(),
                                                                              core_config=self.get_core_config(),
                                                                              target_platform_capabilities=self.get_tpc(),
                                                                              new_experimental_exporter=experimental_exporter
                                                                              )
            ptq_gptq_model, quantization_info = mct.keras_gradient_post_training_quantization_experimental(model_float,
                                                                                             self.representative_data_gen,
                                                                                             gptq_config=self.get_gptq_config(),
                                                                                             target_kpi=self.get_kpi(),
                                                                                             core_config=self.get_core_config(),
                                                                                             target_platform_capabilities=self.get_tpc(),
                                                                                             new_experimental_exporter=experimental_exporter
                                                                                             )
        else:

            ptq_model, quantization_info = mct.keras_post_training_quantization(model_float, representative_data_gen,
                                                                                n_iter=self.num_calibration_iter,
                                                                                quant_config=qc,
                                                                                fw_info=DEFAULT_KERAS_INFO,
                                                                                network_editor=self.get_network_editor(),
                                                                                target_platform_capabilities=tpc)
            ptq_gptq_model, quantization_info = mct.keras_post_training_quantization(model_float, representative_data_gen,
                                                                                     n_iter=self.num_calibration_iter,
                                                                                     quant_config=qc,
                                                                                     fw_info=DEFAULT_KERAS_INFO,
                                                                                     network_editor=self.get_network_editor(),
                                                                                     gptq_config=self.get_gptq_config(),
                                                                                     target_platform_capabilities=tpc)

        self.compare(ptq_model, ptq_gptq_model, input_x=x, quantization_info=quantization_info)


class GradientPTQTest(GradientPTQBaseTest):

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        y = float_model(input_x)
        y_hat = quantized_model(input_x)
        cs = cosine_similarity(y.numpy(), y_hat.numpy())
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check: {cs}')


class GradientPTQWeightsUpdateTest(GradientPTQBaseTest):

    def get_gptq_config(self):
        return model_compression_toolkit.gptq.common.gptq_config.GradientPTQConfig(50,
                                                                                   optimizer=tf.keras.optimizers.Adam(
                                                                                       learning_rate=1e-2),
                                                                                   optimizer_rest=tf.keras.optimizers.Adam(
                                                                                       learning_rate=1e-1),
                                                                                   train_bias=True,
                                                                                   sam_optimization=self.sam_optimization,
                                                                                   loss=multiple_tensors_mse_loss)

    def compare(self, quantized_model, quantized_gptq_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(len(quantized_model.weights) == len(quantized_gptq_model.weights),
                                  msg='float model number of weights different from quantized model: ' +
                                      f'{len(quantized_gptq_model.weights)} != {len(quantized_model.weights)}')
        # check relevant weights were updated
        weights_diff = []
        for l_q, l_f in zip(quantized_model.layers, quantized_gptq_model.layers):
            if self.get_fw_info().get_kernel_op_attributes(type(l_q))[0] is not None:
                for w_q, w_f in zip(l_q.weights, l_f.weights):
                    weights_diff.append(np.any(w_q.numpy() != w_f.numpy()))
        self.unit_test.assertTrue(all(weights_diff), msg="Some weights weren't updated")


class GradientPTQLearnRateZeroTest(GradientPTQBaseTest):

    def get_gptq_config(self):
        return model_compression_toolkit.gptq.common.gptq_config.GradientPTQConfig(1,
                                                                                   optimizer=tf.keras.optimizers.SGD(
                                                                                       learning_rate=0.0),
                                                                                   optimizer_rest=tf.keras.optimizers.SGD(
                                                                                       learning_rate=0.0),
                                                                                   train_bias=True,
                                                                                   sam_optimization=self.sam_optimization,
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


class GradientPTQWeightedLossTest(GradientPTQBaseTest):

    def get_gptq_config(self):
        return model_compression_toolkit.gptq.common.gptq_config.GradientPTQConfig(5,
                                                                                   optimizer=tf.keras.optimizers.Adam(
                                                                                       learning_rate=0.0001),
                                                                                   optimizer_rest=tf.keras.optimizers.Adam(
                                                                                       learning_rate=0.0001),
                                                                                   loss=multiple_tensors_mse_loss,
                                                                                   train_bias=True,
                                                                                   use_jac_based_weights=True,
                                                                                   num_samples_for_loss=16,
                                                                                   norm_weights=False)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check: {cs}')