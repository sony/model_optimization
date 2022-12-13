# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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

import tensorflow as tf
from keras.engine.input_layer import InputLayer
import model_compression_toolkit as mct
from model_compression_toolkit.core.common.constants import LATEST

from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model, \
    generate_mixed_precision_test_tp_model, generate_tp_model_with_activation_mp
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_keras_tpc
from packaging import version

if version.parse(tf.__version__) < version.parse("2.6"):
    from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, Reshape, ZeroPadding2D, \
        Dropout, MaxPooling2D, Activation, ReLU, Add, PReLU, Flatten, Cropping2D, Permute, LeakyReLU, Subtract, Multiply
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Reshape, ZeroPadding2D, \
    Dropout, MaxPooling2D, Activation, ReLU, Add, PReLU, Flatten, Cropping2D, Permute, LeakyReLU, Subtract, Multiply, \
    Conv2DTranspose

tp = mct.target_platform


def get_tpc(name, weight_bits=8, activation_bits=8,
            weights_quantization_method=mct.target_platform.QuantizationMethod.POWER_OF_TWO):
    tp_model = generate_test_tp_model({'weights_n_bits': weight_bits,
                                       'activation_n_bits': activation_bits,
                                       'weights_quantization_method': weights_quantization_method})
    return generate_keras_tpc(name=name, tp_model=tp_model)


def get_16bit_tpc(name):
    tp_model = generate_test_tp_model({'weights_n_bits': 16,
                                       'activation_n_bits': 16})
    return generate_keras_tpc(name=name, tp_model=tp_model)


def get_16bit_tpc_per_tensor(name):
    tp_model = generate_test_tp_model({'weights_n_bits': 16,
                                       'activation_n_bits': 16,
                                       "weights_per_channel_threshold": False})
    return generate_keras_tpc(name=name, tp_model=tp_model)


def get_quantization_disabled_keras_tpc(name):
    tp = generate_test_tp_model({'enable_weights_quantization': False,
                                 'enable_activation_quantization': False})
    return generate_keras_tpc(name=name, tp_model=tp)


def get_activation_quantization_disabled_keras_tpc(name):
    tp = generate_test_tp_model({'enable_activation_quantization': False})
    return generate_keras_tpc(name=name, tp_model=tp)


def get_weights_quantization_disabled_keras_tpc(name):
    tp = generate_test_tp_model({'enable_weights_quantization': False})
    return generate_keras_tpc(name=name, tp_model=tp)


def get_weights_only_mp_tpc_keras(base_config, mp_bitwidth_candidates_list, name):
    mp_tp_model = generate_mixed_precision_test_tp_model(base_cfg=base_config,
                                                         mp_bitwidth_candidates_list=mp_bitwidth_candidates_list)
    return generate_keras_tpc(name=name, tp_model=mp_tp_model)


def get_tpc_with_activation_mp_keras(base_config, mp_bitwidth_candidates_list, name):
    mp_tp_model = generate_tp_model_with_activation_mp(base_cfg=base_config,
                                                       mp_bitwidth_candidates_list=mp_bitwidth_candidates_list)
    return _generate_keras_mp_with_activation_tpc(name=name, tp_model=mp_tp_model)


def _generate_keras_mp_with_activation_tpc(name: str, tp_model: tp.TargetPlatformModel):
    """
    This is a TESTS ONLY method to generate a Keras TPC that supports activation mixed precision.
    It should only be used inside the get_tpc_with_activation_mp_keras method and not as a separate method in
    specific tests.
    """

    keras_tpc = tp.TargetPlatformCapabilities(tp_model, name=name)

    with keras_tpc:
        tp.OperationsSetToLayers("NoQuantization", [Reshape,
                                                    tf.reshape,
                                                    Permute,
                                                    tf.transpose,
                                                    Flatten,
                                                    Cropping2D,
                                                    ZeroPadding2D,
                                                    Dropout,
                                                    MaxPooling2D,
                                                    tf.split,
                                                    tf.quantization.fake_quant_with_min_max_vars,
                                                    tf.math.argmax,
                                                    tf.shape,
                                                    tf.math.equal,
                                                    tf.gather,
                                                    tf.cast,
                                                    tf.compat.v1.gather,
                                                    tf.nn.top_k,
                                                    tf.__operators__.getitem,
                                                    tf.compat.v1.shape])

        tp.OperationsSetToLayers("Conv", [Conv2D,
                                          DepthwiseConv2D,
                                          tf.nn.conv2d,
                                          tf.nn.depthwise_conv2d,
                                          Conv2DTranspose,
                                          tf.nn.conv2d_transpose])
        tp.OperationsSetToLayers("FullyConnected", [Dense])
        tp.OperationsSetToLayers("AnyReLU", [tf.nn.relu,
                                             tf.nn.relu6,
                                             tf.nn.leaky_relu,
                                             ReLU,
                                             LeakyReLU,
                                             tp.LayerFilterParams(Activation, activation="relu"),
                                             tp.LayerFilterParams(Activation, activation="leaky_relu")])
        tp.OperationsSetToLayers("Add", [tf.add, Add])
        tp.OperationsSetToLayers("Sub", [tf.subtract, Subtract])
        tp.OperationsSetToLayers("Mul", [tf.math.multiply, Multiply])
        tp.OperationsSetToLayers("Div", [tf.math.divide])
        tp.OperationsSetToLayers("PReLU", [PReLU])
        tp.OperationsSetToLayers("Swish", [tf.nn.swish, tp.LayerFilterParams(Activation, activation="swish")])
        tp.OperationsSetToLayers("Sigmoid", [tf.nn.sigmoid, tp.LayerFilterParams(Activation, activation="sigmoid")])
        tp.OperationsSetToLayers("Tanh", [tf.nn.tanh, tp.LayerFilterParams(Activation, activation="tanh")])
        tp.OperationsSetToLayers("Input", [InputLayer])

    return keras_tpc
