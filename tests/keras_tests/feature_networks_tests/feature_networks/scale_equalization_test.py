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

import copy
import numpy as np
import tensorflow as tf

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.substitutions.scale_equalization import fixed_second_moment_after_relu, \
    fixed_mean_after_relu
from model_compression_toolkit.core.keras.constants import DEPTHWISE_KERNEL, KERNEL
from model_compression_toolkit.core.keras.default_framework_info import KerasInfo
from tests.keras_tests.tpc_keras import get_16bit_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

keras = tf.keras
layers = keras.layers


def w_init():
    return tf.keras.initializers.RandomUniform(minval=1., maxval=10.)


"""
This test checks the Channel Scale Equalization feature.
"""


class ScaleEqualizationTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, first_op2d, second_op2d, act_node=layers.ReLU(), zero_pad=False):
        self.first_op2d = first_op2d
        self.first_op2d.kernel_initializer = w_init()
        self.act_node = act_node
        self.second_op2d = second_op2d
        self.second_op2d.kernel_initializer = w_init()
        self.zero_pad = zero_pad
        super().__init__(unit_test,
                         input_shape=(16, 16, 3))

    def get_tpc(self):
        return get_16bit_tpc("scale_equalization_bound_test")

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                           relu_bound_to_power_of_2=False, weights_bias_correction=False,
                                           activation_channel_equalization=True)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = self.first_op2d(inputs)
        x = layers.BatchNormalization(gamma_initializer=w_init())(x)
        x = self.act_node(x)
        if self.zero_pad:
            x = layers.ZeroPadding2D()(x)
        outputs = self.second_op2d(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        def _set_new_weights(model):
            # Patch to change the kernel weights since there is a bug with initializer
            # in DepthwiseConv2D (it ignores the min and max values, and set 0 for some of them
            # even though the test requires the weights to be positive.
            weights = copy.deepcopy(model.weights)
            attr = 'depthwise_kernel' if isinstance(self.first_op2d, layers.DepthwiseConv2D) else 'kernel'
            a = 5 if self.zero_pad else 4
            attr2 = 'depthwise_kernel' if isinstance(self.second_op2d, layers.DepthwiseConv2D) else 'kernel'
            new_w = []
            for w in weights:
                if w.name.startswith(model.layers[1].name + '/' + attr):
                    new_w.append(tf.Variable(w + 10.0, name=w.name.replace(':0', "")))
                elif w.name.startswith(model.layers[a].name + '/' + attr2):
                    new_w.append(tf.Variable(w + 10.0, name=w.name.replace(':0', "")))
                else:
                    new_w.append(w)
            model.set_weights(new_w)

        _set_new_weights(model)

        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):

        q_first_linear_op_index = 2
        q_second_linear_op_index = 5 + int(self.zero_pad) + int(
            isinstance(self.first_op2d, layers.Dense))
        f_first_linear_op_index = 1
        f_second_linear_op_index = 4 + int(self.zero_pad)

        first_op_attr = DEPTHWISE_KERNEL if isinstance(self.first_op2d, layers.DepthwiseConv2D) else KERNEL
        second_op_attr = DEPTHWISE_KERNEL if isinstance(self.second_op2d, layers.DepthwiseConv2D) else KERNEL

        quantized_model_layer1_weight = quantized_model.layers[q_first_linear_op_index].get_quantized_weights()[first_op_attr]
        quantized_model_layer2_weight = quantized_model.layers[q_second_linear_op_index].get_quantized_weights()[second_op_attr]

        float_model_layer1_weight = float_model.layers[f_first_linear_op_index].weights[0]
        float_model_layer2_weight = float_model.layers[f_second_linear_op_index].weights[0]

        gamma = np.abs(float_model.layers[f_first_linear_op_index + 1].gamma)
        bn_beta = float_model.layers[f_first_linear_op_index + 1].beta

        fixed_second_moment_vector = fixed_second_moment_after_relu(bn_beta, gamma)
        fixed_mean_vector = fixed_mean_after_relu(bn_beta, gamma)
        fixed_std_vector = np.sqrt(fixed_second_moment_vector - np.power(fixed_mean_vector, 2))

        scale_factor = 1.0 / fixed_std_vector
        scale_factor = np.minimum(scale_factor, 1.0)

        # disable bn folding
        if type(quantized_model.layers[q_first_linear_op_index].layer) == layers.DepthwiseConv2D:
            gamma = gamma.reshape(1, 1, quantized_model_layer1_weight.shape[-2], quantized_model_layer1_weight.shape[-1])
        elif type(quantized_model.layers[q_first_linear_op_index].layer) == layers.Conv2DTranspose:
            gamma = gamma.reshape(1, 1, -1, 1)
        else:
            gamma = gamma.reshape(1, 1, 1, -1)
        quantized_model_layer1_weight_without_bn_fold = quantized_model_layer1_weight / gamma

        if (type(quantized_model.layers[q_first_linear_op_index].layer) == layers.DepthwiseConv2D) \
                or (type(quantized_model.layers[q_second_linear_op_index].layer) == layers.DepthwiseConv2D):
            alpha = np.mean(quantized_model_layer1_weight_without_bn_fold / float_model_layer1_weight)
            beta = np.mean(float_model_layer2_weight / quantized_model_layer2_weight)
            scale_factor = np.mean(scale_factor)
        else:
            first_layer_chn_dim = KerasInfo.get_kernel_channels(
                type(quantized_model.layers[q_first_linear_op_index].layer))[0]
            second_layer_chn_dim = KerasInfo.get_kernel_channels(
                type(quantized_model.layers[q_second_linear_op_index].layer))[1]

            first_layer_axes = tuple(np.delete(np.arange(quantized_model_layer1_weight.numpy().ndim),
                                               first_layer_chn_dim))
            second_layer_axes = tuple(np.delete(np.arange(quantized_model_layer2_weight.numpy().ndim),
                                                second_layer_chn_dim))

            alpha = np.mean(quantized_model_layer1_weight_without_bn_fold / float_model_layer1_weight,
                            axis=first_layer_axes)
            beta = np.mean(float_model_layer2_weight / quantized_model_layer2_weight, axis=second_layer_axes)

        self.unit_test.assertTrue(np.allclose(alpha, beta, atol=1e-1))
        self.unit_test.assertTrue((np.isclose(alpha, 1.0, atol=1e-1) + np.less(alpha, 1.0)).all())
        self.unit_test.assertTrue(np.allclose(alpha, scale_factor, atol=1e-1))
