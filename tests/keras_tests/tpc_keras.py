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

from keras.engine.input_layer import InputLayer
import tensorflow as tf
import model_compression_toolkit as mct

from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_keras_tpc

if tf.__version__ < "2.6":
    from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose, Reshape, ZeroPadding2D, \
        Dropout, \
        MaxPooling2D, Activation, ReLU, Add, PReLU, Flatten, Cropping2D, BatchNormalization
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose, Reshape, ZeroPadding2D, \
        Dropout, MaxPooling2D, Activation, ReLU, Add, PReLU, Flatten, Cropping2D, BatchNormalization

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


def generate_activation_mp_tpc_keras(tp_model, name="activation_mp_keras_tp"):
    ftp_keras = tp.TargetPlatformCapabilities(tp_model,
                                              name=name)
    with ftp_keras:
        tp.OperationsSetToLayers("NoQuantization", [Reshape,
                                                    tf.reshape,
                                                    Flatten,
                                                    Cropping2D,
                                                    ZeroPadding2D,
                                                    Dropout,
                                                    MaxPooling2D,
                                                    tf.split,
                                                    tf.quantization.fake_quant_with_min_max_vars,
                                                    tf.math.argmax,
                                                    tf.shape,
                                                    tf.__operators__.getitem])

        tp.OperationsSetToLayers("Weights_n_Activation", [Conv2D,
                                                          DepthwiseConv2D,
                                                          tf.nn.conv2d,
                                                          tf.nn.depthwise_conv2d,
                                                          Dense,
                                                          Conv2DTranspose,
                                                          tf.nn.conv2d_transpose])

        tp.OperationsSetToLayers("Activation", [tf.nn.relu,
                                                tf.nn.relu6,
                                                tp.LayerFilterParams(ReLU, negative_slope=0.0),
                                                tp.LayerFilterParams(Activation, activation="relu"),
                                                tf.add,
                                                Add,
                                                PReLU,
                                                tf.nn.swish,
                                                tp.LayerFilterParams(Activation, activation="swish"),
                                                tf.nn.sigmoid,
                                                tp.LayerFilterParams(Activation, activation="sigmoid"),
                                                tf.nn.tanh,
                                                tp.LayerFilterParams(Activation, activation="tanh"),
                                                InputLayer,
                                                BatchNormalization])

    return ftp_keras
