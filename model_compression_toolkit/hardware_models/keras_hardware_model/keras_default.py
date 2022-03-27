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

import tensorflow as tf

if tf.__version__ < "2.6":
    from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, Activation, ReLU, Add, PReLU, Flatten, Cropping2D, BatchNormalization
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose, Reshape, ZeroPadding2D, \
    Dropout, MaxPooling2D, Activation, ReLU, Add, PReLU, Flatten, Cropping2D, BatchNormalization

from model_compression_toolkit.hardware_models.default_hwm import get_default_hardware_model

import model_compression_toolkit as mct
hwm = mct.hardware_representation


def get_default_hwm_keras():
    default_hwm = get_default_hardware_model()
    default_hwm_keras = hwm.FrameworkHardwareModel(default_hwm,
                                                   name='default_hwm_keras')

    with default_hwm_keras:
        hwm.OperationsSetToLayers("NoQuantization", [Reshape,
                                                     tf.reshape,
                                                     Flatten,
                                                     Cropping2D,
                                                     ZeroPadding2D,
                                                     Dropout,
                                                     MaxPooling2D,
                                                     tf.split,
                                                     tf.quantization.fake_quant_with_min_max_vars,
                                                     BatchNormalization])

        hwm.OperationsSetToLayers("Conv", [Conv2D,
                                           DepthwiseConv2D,
                                           tf.nn.conv2d,
                                           tf.nn.depthwise_conv2d])

        hwm.OperationsSetToLayers("FullyConnected", [Dense])

        hwm.OperationsSetToLayers("ConvTranspose", [Conv2DTranspose,
                                                    tf.nn.conv2d_transpose])

        hwm.OperationsSetToLayers("AnyReLU", [tf.nn.relu,
                                              tf.nn.relu6,
                                              hwm.LayerFilterParams(ReLU, negative_slope=0.0),
                                              hwm.LayerFilterParams(Activation, activation="relu")])

        hwm.OperationsSetToLayers("Add", [tf.add,
                                          Add])

        hwm.OperationsSetToLayers("PReLU", [PReLU])

        hwm.OperationsSetToLayers("Swish", [tf.nn.swish,
                                            hwm.LayerFilterParams(Activation, activation="swish")])

        hwm.OperationsSetToLayers("Sigmoid", [tf.nn.sigmoid,
                                              hwm.LayerFilterParams(Activation, activation="sigmoid")])

        hwm.OperationsSetToLayers("Tanh", [tf.nn.tanh,
                                           hwm.LayerFilterParams(Activation, activation="tanh")])

    return default_hwm_keras


