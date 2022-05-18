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

from model_compression_toolkit.common.target_platform import TargetPlatformModel

if tf.__version__ < "2.6":
    from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, Activation, ReLU, Add, PReLU, Flatten, Cropping2D
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose, Reshape, ZeroPadding2D, \
    Dropout, MaxPooling2D, Activation, ReLU, Add, PReLU, Flatten, Cropping2D

from model_compression_toolkit.tpc_models.default_tp_model import get_default_tp_model

import model_compression_toolkit as mct
tpc = mct.target_platform


def get_default_keras_tpc():
    default_tp_model = get_default_tp_model()
    return generate_keras_default_tpc(name='default_keras_tpc',
                                      tp_model=default_tp_model)


def generate_keras_default_tpc(name: str, tp_model: TargetPlatformModel):
    """
    Generates a TargetPlatformCapabilities object with default operation sets to layers mapping.

    Args:
        name: Name of the TargetPlatformCapabilities.
        tp_model: TargetPlatformModel object.

    Returns: a TargetPlatformCapabilities object for the given TargetPlatformModel.
    """

    keras_tpc = tpc.TargetPlatformCapabilities(tp_model,
                                               name=name)
    with keras_tpc:
        tpc.OperationsSetToLayers("NoQuantization", [Reshape,
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
                                                     tf.__operators__.getitem,
                                                     tf.compat.v1.shape])

        tpc.OperationsSetToLayers("Conv", [Conv2D,
                                           DepthwiseConv2D,
                                           tf.nn.conv2d,
                                           tf.nn.depthwise_conv2d])

        tpc.OperationsSetToLayers("FullyConnected", [Dense])

        tpc.OperationsSetToLayers("ConvTranspose", [Conv2DTranspose,
                                                    tf.nn.conv2d_transpose])

        tpc.OperationsSetToLayers("AnyReLU", [tf.nn.relu,
                                              tf.nn.relu6,
                                              tpc.LayerFilterParams(ReLU, negative_slope=0.0),
                                              tpc.LayerFilterParams(Activation, activation="relu")])

        tpc.OperationsSetToLayers("Add", [tf.add,
                                          Add])

        tpc.OperationsSetToLayers("PReLU", [PReLU])

        tpc.OperationsSetToLayers("Swish", [tf.nn.swish,
                                            tpc.LayerFilterParams(Activation, activation="swish")])

        tpc.OperationsSetToLayers("Sigmoid", [tf.nn.sigmoid,
                                              tpc.LayerFilterParams(Activation, activation="sigmoid")])

        tpc.OperationsSetToLayers("Tanh", [tf.nn.tanh,
                                           tpc.LayerFilterParams(Activation, activation="tanh")])
    return keras_tpc
