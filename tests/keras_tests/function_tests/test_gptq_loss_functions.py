# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from functools import partial

import numpy as np
from model_compression_toolkit import get_keras_gptq_config, keras_post_training_quantization, keras_gradient_post_training_quantization_experimental, \
    QuantizationConfig, QuantizationErrorMethod, GradientPTQConfig, RoundingType, CoreConfig
import tensorflow as tf
from model_compression_toolkit.gptq.keras.gptq_loss import multiple_tensors_mse_loss, GPTQMultipleTensorsLoss
import model_compression_toolkit as mct

layers = tf.keras.layers


class TestGPTQLossFunctions(unittest.TestCase):
    SHAPE = [1, 16, 16, 3]

    def _build_model(self) -> tf.keras.Model:
        """
        This function generate a simple network to test GPTQ loss functions
        Args:
            in_input_shape: Input shape list

        Returns:

        """
        inputs = layers.Input(shape=self.SHAPE[1:])
        x1 = layers.Conv2D(3, 4, use_bias=False)(inputs)
        x = layers.ReLU()(x1)
        x2 = layers.Conv2D(7, 8, use_bias=False)(x)
        model = tf.keras.Model(inputs=inputs, outputs=[x1, x2])
        return model

    def _random_datagen(self):
        for _ in range(10):
            yield [np.random.random(self.SHAPE)]

    @staticmethod
    def _compute_gradients(loss_fn, fxp_model, input_data, in_y_float):
        with tf.GradientTape(persistent=True) as tape:
            y_fxp = fxp_model(input_data, training=True)  # running fxp model
            loss_value = loss_fn(y_fxp, in_y_float)
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, fxp_model.trainable_weights)
        return loss_value, grads

    def _train(self, float_model, quantized_model, loss_fn):
        in_optimizer = tf.keras.optimizers.SGD(learning_rate=20.0)
        for input_data in self._random_datagen():
            y_float = float_model(input_data)
            # run quantized model and calculate loss & gradients
            loss_value_step, grads = self._compute_gradients(loss_fn, quantized_model, input_data, y_float)
            in_optimizer.apply_gradients(zip(grads, quantized_model.trainable_weights))

    def _compare(self, original_weights, trained_weights):
        self.assertTrue(all([np.mean(o != t) > 0.9 for o, t in zip(original_weights, trained_weights)]))

    def _init_test(self):
        float_model = self._build_model()
        quantized_model = self._build_model()
        original_weights = [w.numpy() for w in quantized_model.trainable_weights]
        return float_model, quantized_model, original_weights

    def _run_and_compare(self, float_model, quantized_model, loss_fn, original_weights):
        self._train(float_model, quantized_model, loss_fn)
        trained_weights = [w.numpy() for w in quantized_model.trainable_weights]
        self._compare(original_weights, trained_weights)

    def test_mse_loss(self):
        float_model, quantized_model, original_weights = self._init_test()
        loss_fn = partial(multiple_tensors_mse_loss, fxp_w_list=None, flp_w_list=None, act_bn_mean=None, act_bn_std=None)
        self._run_and_compare(float_model, quantized_model, loss_fn, original_weights)

    def test_weighted_mse_loss(self):
        float_model, quantized_model, original_weights = self._init_test()
        loss_fn = partial(multiple_tensors_mse_loss, fxp_w_list=None, flp_w_list=None, act_bn_mean=None, act_bn_std=None, loss_weights=[0.9, 1.1])
        self._run_and_compare(float_model, quantized_model, loss_fn, original_weights)

    def test_activation_mse_loss(self):
        float_model, quantized_model, original_weights = self._init_test()
        loss = GPTQMultipleTensorsLoss(norm_loss=False)
        loss_fn = partial(loss.__call__, fxp_w_list=None, flp_w_list=None, act_bn_mean=None, act_bn_std=None, weights_for_average_loss=None)
        self._run_and_compare(float_model, quantized_model, loss_fn, original_weights)

    def test_weighted_activation_mse_loss(self):
        float_model, quantized_model, original_weights = self._init_test()
        loss = GPTQMultipleTensorsLoss(norm_loss=False)
        loss_fn = partial(loss.__call__, fxp_w_list=None, flp_w_list=None, act_bn_mean=None, act_bn_std=None, weights_for_average_loss=[0.9, 1.1])
        self._run_and_compare(float_model, quantized_model, loss_fn, original_weights)


if __name__ == '__main__':
    unittest.main()
