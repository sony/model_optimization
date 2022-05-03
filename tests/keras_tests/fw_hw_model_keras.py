# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
from model_compression_toolkit.tpc_models.keras_tp_models.keras_default import generate_keras_default_tpc
from tests.common_tests.helpers.generate_test_hw_model import generate_test_hw_model

if tf.__version__ < "2.6":
    from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, Activation, ReLU, Add, PReLU, Flatten, Cropping2D, BatchNormalization
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose, Reshape, ZeroPadding2D, \
    Dropout, MaxPooling2D, Activation, ReLU, Add, PReLU, Flatten, Cropping2D, BatchNormalization


hwm = mct.target_platform


def get_16bit_fw_hw_model(name):
    hw_model = generate_test_hw_model({'weights_n_bits': 16,
                                       'activation_n_bits': 16})
    return generate_keras_default_tpc(name=name, tp_model=hw_model)


def get_quantization_disabled_keras_hw_model(name):
    hwm = generate_test_hw_model({'enable_weights_quantization': False,
                                  'enable_activation_quantization': False})
    return generate_keras_default_tpc(name=name, tp_model=hwm)


def generate_activation_mp_fhw_model_keras(hardware_model, name="activation_mp_keras_hwm"):

    fhwm_keras = hwm.TargetPlatformCapabilities(hardware_model,
                                                name=name)
    with fhwm_keras:
        hwm.OperationsSetToLayers("NoQuantization", [Reshape,
                                                     tf.reshape,
                                                     Flatten,
                                                     Cropping2D,
                                                     ZeroPadding2D,
                                                     Dropout,
                                                     MaxPooling2D,
                                                     tf.split,
                                                     tf.quantization.fake_quant_with_min_max_vars])

        hwm.OperationsSetToLayers("Weights_n_Activation", [Conv2D,
                                                           DepthwiseConv2D,
                                                           tf.nn.conv2d,
                                                           tf.nn.depthwise_conv2d,
                                                           Dense,
                                                           Conv2DTranspose,
                                                           tf.nn.conv2d_transpose])

        hwm.OperationsSetToLayers("Activation", [tf.nn.relu,
                                                 tf.nn.relu6,
                                                 hwm.LayerFilterParams(ReLU, negative_slope=0.0),
                                                 hwm.LayerFilterParams(Activation, activation="relu"),
                                                 tf.add,
                                                 Add,
                                                 PReLU,
                                                 tf.nn.swish,
                                                 hwm.LayerFilterParams(Activation, activation="swish"),
                                                 tf.nn.sigmoid,
                                                 hwm.LayerFilterParams(Activation, activation="sigmoid"),
                                                 tf.nn.tanh,
                                                 hwm.LayerFilterParams(Activation, activation="tanh"),
                                                 InputLayer,
                                                 BatchNormalization])

    return fhwm_keras