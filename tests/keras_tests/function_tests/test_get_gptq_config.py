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
import unittest
from typing import List

import numpy as np

import model_compression_toolkit as mct
from mct_quantizers import QuantizationMethod

from model_compression_toolkit.gptq import get_keras_gptq_config, keras_gradient_post_training_quantization, GradientPTQConfig, RoundingType
from model_compression_toolkit.core import QuantizationConfig, QuantizationErrorMethod, CoreConfig
from model_compression_toolkit import DefaultDict
import tensorflow as tf

from model_compression_toolkit.gptq.common.gptq_config import GPTQHessianScoresConfig
from model_compression_toolkit.gptq.common.gptq_constants import QUANT_PARAM_LEARNING_STR, MAX_LSB_STR
from model_compression_toolkit.gptq.keras.gptq_loss import multiple_tensors_mse_loss
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from tests.common_tests.helpers.generate_test_tpc import generate_test_tpc

layers = tf.keras.layers
SHAPE = [1, 16, 16, 3]


def build_model(in_input_shape: List[int]) -> tf.keras.Model:
    """
    This function generate a simple network to test GPTQ
    Args:
        in_input_shape: Input shape list

    Returns:

    """
    inputs = layers.Input(shape=in_input_shape)
    x = layers.Conv2D(3, 4)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.Conv2D(7, 8)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.ReLU()(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def random_datagen():
    for _ in range(20):
        yield [np.random.random(SHAPE)]


class TestGetGPTQConfig(unittest.TestCase):
    def setUp(self):
        self.qc = QuantizationConfig(QuantizationErrorMethod.MSE, QuantizationErrorMethod.MSE,
                                     weights_bias_correction=False)  # disable bias correction when working with GPTQ
        self.cc = CoreConfig(quantization_config=self.qc)

        default_reg_factor = 1
        default_hessian_cfg = GPTQHessianScoresConfig(per_sample=False, hessians_num_samples=2)
        test_hessian_weights_config = GPTQHessianScoresConfig(per_sample=False,
                                                              hessians_num_samples=2,
                                                              norm_scores=False,
                                                              log_norm=True,
                                                              scale_log_norm=True)

        self.gptq_configurations = [GradientPTQConfig(1, optimizer=tf.keras.optimizers.RMSprop(),
                                                      optimizer_rest=tf.keras.optimizers.RMSprop(),
                                                      train_bias=True,
                                                      loss=multiple_tensors_mse_loss,
                                                      rounding_type=RoundingType.SoftQuantizer,
                                                      hessian_weights_config=default_hessian_cfg,
                                                      regularization_factor=default_reg_factor,
                                                      gradual_activation_quantization_config=None),
                                    GradientPTQConfig(1, optimizer=tf.keras.optimizers.Adam(),
                                                      optimizer_rest=tf.keras.optimizers.Adam(),
                                                      train_bias=True,
                                                      loss=multiple_tensors_mse_loss,
                                                      rounding_type=RoundingType.SoftQuantizer,
                                                      hessian_weights_config=default_hessian_cfg,
                                                      regularization_factor=default_reg_factor,
                                                      gradual_activation_quantization_config=None),
                                    GradientPTQConfig(1, optimizer=tf.keras.optimizers.Adam(),
                                                      optimizer_rest=tf.keras.optimizers.Adam(),
                                                      train_bias=True,
                                                      loss=multiple_tensors_mse_loss,
                                                      rounding_type=RoundingType.SoftQuantizer,
                                                      hessian_weights_config=default_hessian_cfg,
                                                      regularization_factor=15,
                                                      gradual_activation_quantization_config=None),
                                    GradientPTQConfig(1, optimizer=tf.keras.optimizers.Adam(),
                                                      optimizer_rest=tf.keras.optimizers.Adam(),
                                                      train_bias=True,
                                                      loss=multiple_tensors_mse_loss,
                                                      rounding_type=RoundingType.SoftQuantizer,
                                                      hessian_weights_config=default_hessian_cfg,
                                                      regularization_factor=default_reg_factor,
                                                      gradual_activation_quantization_config=None,
                                                      gptq_quantizer_params_override={QUANT_PARAM_LEARNING_STR: True}),
                                    GradientPTQConfig(1, optimizer=tf.keras.optimizers.Adam(),
                                                      optimizer_rest=tf.keras.optimizers.Adam(),
                                                      train_bias=True,
                                                      loss=multiple_tensors_mse_loss,
                                                      rounding_type=RoundingType.SoftQuantizer,
                                                      hessian_weights_config=test_hessian_weights_config,
                                                      regularization_factor=default_reg_factor,
                                                      gradual_activation_quantization_config=None),
                                    GradientPTQConfig(1, optimizer=tf.keras.optimizers.Adam(),
                                                      optimizer_rest=tf.keras.optimizers.Adam(),
                                                      train_bias=True,
                                                      loss=multiple_tensors_mse_loss,
                                                      rounding_type=RoundingType.STE,
                                                      hessian_weights_config=default_hessian_cfg,
                                                      regularization_factor=default_reg_factor,
                                                      gradual_activation_quantization_config=None,
                                                      gptq_quantizer_params_override={
                                                          MAX_LSB_STR: DefaultDict(default_value=1)}),
                                    get_keras_gptq_config(n_epochs=1,
                                                          optimizer=tf.keras.optimizers.Adam()),
                                    get_keras_gptq_config(n_epochs=1,
                                                          optimizer=tf.keras.optimizers.Adam(),
                                                          regularization_factor=0.001)]

        pot_tp = generate_test_tpc({'weights_quantization_method': QuantizationMethod.POWER_OF_TWO})
        self.pot_weights_tpc = generate_keras_tpc(name="gptq_pot_config_test", tpc=pot_tp)

        symmetric_tp = generate_test_tpc({'weights_quantization_method': QuantizationMethod.SYMMETRIC})
        self.symmetric_weights_tpc = generate_keras_tpc(name="gptq_symmetric_config_test", tpc=symmetric_tp)

    def test_get_keras_gptq_config_pot(self):
        # This call removes the effect of @tf.function decoration and executes the decorated function eagerly, which
        # enabled tracing for code coverage.
        tf.config.run_functions_eagerly(True)
        for i, gptq_config in enumerate(self.gptq_configurations):
            # Reducing the default number of samples for GPTQ Hessian approximation
            # to allow quick execution of the test
            gptq_config.hessian_weights_config.hessians_num_samples = 2

            keras_gradient_post_training_quantization(in_model=build_model(SHAPE[1:]),
                                                      representative_data_gen=random_datagen,
                                                      gptq_config=gptq_config, core_config=self.cc,
                                                      target_platform_capabilities=self.pot_weights_tpc)

        tf.config.run_functions_eagerly(False)

    def test_get_keras_gptq_config_symmetric(self):
        # This call removes the effect of @tf.function decoration and executes the decorated function eagerly, which
        # enabled tracing for code coverage.
        tf.config.run_functions_eagerly(True)

        for i, gptq_config in enumerate(self.gptq_configurations):
            # Reducing the default number of samples for GPTQ Hessian approximation
            # to allow quick execution of the test
            gptq_config.hessian_weights_config.hessians_num_samples = 2

            keras_gradient_post_training_quantization(in_model=build_model(SHAPE[1:]),
                                                      representative_data_gen=random_datagen,
                                                      gptq_config=gptq_config, core_config=self.cc,
                                                      target_platform_capabilities=self.symmetric_weights_tpc)

        tf.config.run_functions_eagerly(False)

    def test_gradual_activation_quantization_custom_config(self):
        custom_config = mct.gptq.GradualActivationQuantizationConfig(
            q_fraction_scheduler_policy=mct.gptq.QFractionLinearAnnealingConfig(initial_q_fraction=0.2,
                                                                                target_q_fraction=0.8,
                                                                                start_step=1,
                                                                                end_step=2)
        )
        config = get_keras_gptq_config(n_epochs=5, gradual_activation_quantization=custom_config)
        self.assertEqual(config.gradual_activation_quantization_config, custom_config)

    def test_gradual_activation_quantization_invalid_type(self):
        with self.assertRaises(TypeError):
            get_keras_gptq_config(n_epochs=5, gradual_activation_quantization='invalid_type')


if __name__ == '__main__':
    unittest.main()
