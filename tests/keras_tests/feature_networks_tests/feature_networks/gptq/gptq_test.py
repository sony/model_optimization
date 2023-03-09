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
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig, RoundingType, GradientPTQConfigV2
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.gptq.keras.gptq_loss import multiple_tensors_mse_loss
from tests.common_tests.helpers.tensors_compare import cosine_similarity
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.keras_tests.tpc_keras import get_16bit_tpc, get_tpc

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
    x = layers.Conv2D(64, 4, bias_initializer='glorot_uniform')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.Conv2D(64, 8, bias_initializer='glorot_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(20)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


class GradientPTQBaseTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, quant_method=QuantizationMethod.SYMMETRIC, rounding_type=RoundingType.STE,
                 per_channel=True, input_shape=(1, 16, 16, 3),
                 hessian_weights=True, log_norm_weights=True):
        super().__init__(unit_test,
                         input_shape=input_shape)

        self.quant_method = quant_method
        self.rounding_type = rounding_type
        self.per_channel = per_channel
        self.hessian_weights = hessian_weights
        self.log_norm_weights = log_norm_weights

    def get_tpc(self):
        return get_tpc("gptq_test", 16, 16, self.quant_method)

    def get_quantization_config(self):
        return mct.QuantizationConfig(activation_error_method=mct.QuantizationErrorMethod.NOCLIPPING,
                                      weights_error_method=mct.QuantizationErrorMethod.NOCLIPPING,
                                      relu_bound_to_power_of_2=True,
                                      weights_bias_correction=False,
                                      weights_per_channel_threshold=self.per_channel)

    def get_gptq_config(self):
        return GradientPTQConfig(5,
                                 optimizer=tf.keras.optimizers.Adam(
                                     learning_rate=0.0001),
                                 optimizer_rest=tf.keras.optimizers.Adam(
                                     learning_rate=0.0001),
                                 loss=multiple_tensors_mse_loss,
                                 rounding_type=self.rounding_type,
                                 train_bias=True,
                                 use_jac_based_weights=self.hessian_weights,
                                 log_norm=self.log_norm_weights)

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
            core_config = self.get_core_config()
            ptq_model, quantization_info = mct.keras_post_training_quantization_experimental(
                model_float,
                self.representative_data_gen_experimental,
                target_kpi=self.get_kpi(),
                core_config=core_config,
                target_platform_capabilities=self.get_tpc(),
                new_experimental_exporter=experimental_exporter
            )
            ptq_gptq_model, quantization_info = mct.keras_gradient_post_training_quantization_experimental(
                model_float,
                self.representative_data_gen_experimental,
                gptq_config=GradientPTQConfigV2.from_v1(self.num_calibration_iter, self.get_gptq_config()),
                target_kpi=self.get_kpi(),
                core_config=core_config,
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
        self.unit_test.assertTrue(np.isclose(cs, 1, rtol=1e-4), msg=f'fail cosine similarity check: {cs}')


class GradientPTQNoTempLearningTest(GradientPTQBaseTest):

    def get_gptq_config(self):
        return GradientPTQConfig(1,
                                 optimizer=tf.keras.optimizers.Adam(
                                     learning_rate=0.0001),
                                 optimizer_rest=tf.keras.optimizers.Adam(
                                     learning_rate=0.0001),
                                 loss=multiple_tensors_mse_loss,
                                 rounding_type=self.rounding_type,
                                 train_bias=True)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        y = float_model(input_x)
        y_hat = quantized_model(input_x)
        cs = cosine_similarity(y.numpy(), y_hat.numpy())
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check: {cs}')


class GradientPTQWeightsUpdateTest(GradientPTQBaseTest):

    def get_gptq_config(self):
        return GradientPTQConfig(50,
                                 optimizer=tf.keras.optimizers.Adam(
                                     learning_rate=1e-2),
                                 optimizer_rest=tf.keras.optimizers.Adam(
                                     learning_rate=1e-1),
                                 train_bias=True,
                                 loss=multiple_tensors_mse_loss,
                                 rounding_type=self.rounding_type)

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

        # Verify that the majority of layers' weights were updated
        self.unit_test.assertTrue(len([b for b in weights_diff if not b]) <= 1, msg="Some weights weren't updated")


class GradientPTQLearnRateZeroTest(GradientPTQBaseTest):

    def get_gptq_config(self):
        return GradientPTQConfig(1,
                                 optimizer=tf.keras.optimizers.SGD(
                                     learning_rate=0.0),
                                 optimizer_rest=tf.keras.optimizers.SGD(
                                     learning_rate=0.0),
                                 train_bias=True,
                                 loss=multiple_tensors_mse_loss,
                                 rounding_type=self.rounding_type)

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
        return GradientPTQConfig(5,
                                 optimizer=tf.keras.optimizers.Adam(
                                     learning_rate=0.0001),
                                 optimizer_rest=tf.keras.optimizers.Adam(
                                     learning_rate=0.0001),
                                 loss=multiple_tensors_mse_loss,
                                 train_bias=True,
                                 use_jac_based_weights=True,
                                 num_samples_for_loss=16,
                                 norm_weights=False,
                                 rounding_type=self.rounding_type)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check: {cs}')


class GradientPTQWithDepthwiseTest(GradientPTQTest):

    def __init__(self, unit_test, rounding_type):
        super().__init__(unit_test, rounding_type=rounding_type,
                         input_shape=(16, 16, 3))

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.DepthwiseConv2D(4)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.DepthwiseConv2D(4)(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model
