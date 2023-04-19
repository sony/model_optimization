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
from packaging import version

if version.parse(tf.__version__) < version.parse("2.6"):
    from tensorflow.keras.layers import Conv2D, Dense, Reshape, ZeroPadding2D, AveragePooling2D, Activation, \
        DepthwiseConv2D, MaxPooling2D, ReLU, Add, Softmax, Concatenate, Multiply, Maximum, Minimum, BatchNormalization
else:
    from keras.layers import Conv2D, Dense, Reshape, ZeroPadding2D, AveragePooling2D, Activation, DepthwiseConv2D, \
        MaxPooling2D, ReLU, Add, Softmax, Concatenate, Multiply, Maximum, Minimum, BatchNormalization

from tensorflow.python.keras.layers.core import SlicingOpLambda
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework.attribute_filter import Eq

from model_compression_toolkit.target_platform_capabilities.tpc_models.tflite_tpc.v1.tp_model import get_tp_model
import model_compression_toolkit as mct
from model_compression_toolkit.target_platform_capabilities.tpc_models.tflite_tpc.v1 import __version__ as TPC_VERSION

tp = mct.target_platform


def get_keras_tpc() -> tp.TargetPlatformCapabilities:
    """
    get a Keras TargetPlatformCapabilities object with default operation sets to layers mapping.
    Returns: a Keras TargetPlatformCapabilities object for the given TargetPlatformModel.
    """
    tflite_tp_model = get_tp_model()
    return generate_keras_tpc(name='tflite_keras', tp_model=tflite_tp_model)


def generate_keras_tpc(name: str, tp_model: tp.TargetPlatformModel):
    """
    Generates a TargetPlatformCapabilities object with default operation sets to layers mapping.

    Args:
        name: Name of the TargetPlatformCapabilities.
        tp_model: TargetPlatformModel object.

    Returns: a TargetPlatformCapabilities object for the given TargetPlatformModel.
    """

    keras_tpc = tp.TargetPlatformCapabilities(tp_model,
                                              name=name,
                                              version=TPC_VERSION)

    with keras_tpc:
        tp.OperationsSetToLayers("NoQuantization", [AveragePooling2D,
                                                                tf.nn.avg_pool2d,
                                                                Concatenate,
                                                                tf.concat,
                                                                MaxPooling2D,
                                                                Multiply,
                                                                tf.multiply,
                                                                Reshape,
                                                                tf.reshape,
                                                                tp.LayerFilterParams(tf.image.resize,
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

        tp.OperationsSetToLayers("FullyConnected", [Dense])
        tp.OperationsSetToLayers("L2Normalization", [tf.math.l2_normalize])
        tp.OperationsSetToLayers("LogSoftmax", [tf.nn.log_softmax])
        tp.OperationsSetToLayers("Tanh", [tf.nn.tanh, tp.LayerFilterParams(Activation, activation="tanh")])
        tp.OperationsSetToLayers("Softmax", [tf.nn.softmax,
                                             Softmax,
                                             tp.LayerFilterParams(Activation, activation="softmax")])
        tp.OperationsSetToLayers("Logistic", [tf.sigmoid, tp.LayerFilterParams(Activation, activation="sigmoid")])

        tp.OperationsSetToLayers("Conv2d", [Conv2D, DepthwiseConv2D])
        tp.OperationsSetToLayers("Relu", [tf.nn.relu,
                                          tf.nn.relu6,
                                          tp.LayerFilterParams(ReLU, Eq("max_value", None) | Eq("max_value", 6)),
                                          tp.LayerFilterParams(Activation, activation="relu")])
        tp.OperationsSetToLayers("Elu", [tf.nn.elu, tp.LayerFilterParams(Activation, activation="elu")])
        tp.OperationsSetToLayers("BatchNorm", [BatchNormalization, tf.nn.batch_normalization])
        tp.OperationsSetToLayers("Squeeze", [tf.squeeze])
        tp.OperationsSetToLayers("BiasAdd", [tf.nn.bias_add])
        tp.OperationsSetToLayers("Add", [tf.add, Add])

    return keras_tpc
