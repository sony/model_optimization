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
    from tensorflow.keras.layers import Conv2D, Dense, Reshape, ZeroPadding2D, AveragePooling2D, Activation, DepthwiseConv2D, MaxPooling2D, ReLU, Add, Softmax, Concatenate, Multiply, Maximum, Minimum, BatchNormalization
else:
    from keras.layers import Conv2D, Dense, Reshape, ZeroPadding2D, AveragePooling2D, Activation, DepthwiseConv2D, MaxPooling2D, ReLU, Add, Softmax, Concatenate, Multiply, Maximum, Minimum, BatchNormalization


from tensorflow.python.keras.layers.core import SlicingOpLambda
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from model_compression_toolkit.common.target_platform import TargetPlatformCapabilities
from model_compression_toolkit.common.target_platform.targetplatform2framework import OperationsSetToLayers, \
    LayerFilterParams
from model_compression_toolkit.common.target_platform.targetplatform2framework.attribute_filter import Eq
from model_compression_toolkit.tpc_models.tflite import get_tflite_tp_model


def get_keras_tp_model_tflite():
    tflite_tp_model = get_tflite_tp_model()
    tflite_keras = TargetPlatformCapabilities(tflite_tp_model, name='tflite_keras')

    with tflite_keras:
        OperationsSetToLayers("PreserveQuantizationParams", [AveragePooling2D,
                                                             tf.nn.avg_pool2d,
                                                             Concatenate,
                                                             tf.concat,
                                                             MaxPooling2D,
                                                             Multiply,
                                                             tf.multiply,
                                                             Reshape,
                                                             tf.reshape,
                                                             LayerFilterParams(tf.image.resize,
                                                                               method=ResizeMethod.BILINEAR),
                                                             tf.nn.space_to_depth,
                                                             ZeroPadding2D,
                                                             tf.gather,
                                                             tf.compat.v1.batch_to_space_nd,
                                                             tf.space_to_batch_nd,
                                                             tf.transpose,
                                                             tf.maximum,
                                                             Maximum,
                                                             tf.minimum,
                                                             Minimum,
                                                             tf.pad,
                                                             tf.slice,
                                                             SlicingOpLambda])

        OperationsSetToLayers("FullyConnected", [Dense])
        OperationsSetToLayers("L2Normalization", [tf.math.l2_normalize])
        OperationsSetToLayers("LogSoftmax", [tf.nn.log_softmax])
        OperationsSetToLayers("Tanh", [tf.nn.tanh,
                                       LayerFilterParams(Activation, activation="tanh")])

        OperationsSetToLayers("Softmax", [tf.nn.softmax,
                                          Softmax,
                                          LayerFilterParams(Activation, activation="softmax")])

        OperationsSetToLayers("Logistic", [tf.sigmoid,
                                           LayerFilterParams(Activation, activation="sigmoid")])

        OperationsSetToLayers("Conv2d", [Conv2D])
        OperationsSetToLayers("DepthwiseConv2D", [DepthwiseConv2D])

        OperationsSetToLayers("Relu", [tf.nn.relu,
                                       tf.nn.relu6,
                                       LayerFilterParams(ReLU, Eq("max_value", None) | Eq("max_value", 6)),
                                       LayerFilterParams(Activation, activation="relu")])

        OperationsSetToLayers("Elu", [tf.nn.elu,
                                      LayerFilterParams(Activation, activation="elu")])

        OperationsSetToLayers("BatchNorm", [BatchNormalization,
                                            tf.nn.batch_normalization])

        OperationsSetToLayers("Squeeze", [tf.squeeze])
        OperationsSetToLayers("BiasAdd", [tf.nn.bias_add])
        OperationsSetToLayers("Add", [tf.add,
                                      Add])

    return tflite_keras

