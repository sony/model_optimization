# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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

from model_compression_toolkit.defaultdict import DefaultDict
from model_compression_toolkit.constants import FOUND_SONY_CUSTOM_LAYERS
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, KERAS_DEPTHWISE_KERNEL, \
    KERAS_KERNEL, BIAS_ATTR, BIAS

if FOUND_SONY_CUSTOM_LAYERS:
    from sony_custom_layers.keras.object_detection.ssd_post_process import SSDPostProcess

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers import Conv2D, DepthwiseConv2D, Dense, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, Activation, ReLU, Add, Subtract, Multiply, PReLU, Flatten, Cropping2D, LeakyReLU, Permute, \
        Conv2DTranspose, Identity
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, Activation, ReLU, Add, Subtract, Multiply, PReLU, Flatten, Cropping2D, LeakyReLU, Permute, \
        Conv2DTranspose, Identity

from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v2.tp_model import get_tp_model
import model_compression_toolkit as mct
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v2 import __version__ as TPC_VERSION

tp = mct.target_platform


def get_keras_tpc() -> tp.TargetPlatformCapabilities:
    """
    get a Keras TargetPlatformCapabilities object with default operation sets to layers mapping.

    Returns: a Keras TargetPlatformCapabilities object for the given TargetPlatformModel.
    """
    imx500_tpc_tp_model = get_tp_model()
    return generate_keras_tpc(name='imx500_tpc_keras_tpc', tp_model=imx500_tpc_tp_model)


def generate_keras_tpc(name: str, tp_model: tp.TargetPlatformModel):
    """
    Generates a TargetPlatformCapabilities object with default operation sets to layers mapping.

    Args:
        name: Name of the TargetPlatformCapabilities.
        tp_model: TargetPlatformModel object.

    Returns: a TargetPlatformCapabilities object for the given TargetPlatformModel.
    """

    keras_tpc = tp.TargetPlatformCapabilities(tp_model, name=name, version=TPC_VERSION)

    no_quant_list = [Identity,
                     tf.identity,
                     Reshape,
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
                     tf.unstack,
                     tf.compat.v1.gather,
                     tf.nn.top_k,
                     tf.__operators__.getitem,
                     tf.image.combined_non_max_suppression,
                     tf.compat.v1.shape]

    if FOUND_SONY_CUSTOM_LAYERS:
        no_quant_list.append(SSDPostProcess)

    with keras_tpc:
        tp.OperationsSetToLayers("NoQuantization", no_quant_list)
        tp.OperationsSetToLayers("Conv",
                                 [Conv2D,
                                  DepthwiseConv2D,
                                  Conv2DTranspose,
                                  tf.nn.conv2d,
                                  tf.nn.depthwise_conv2d,
                                  tf.nn.conv2d_transpose],
                                 # we provide attributes mapping that maps each layer type in the operations set
                                 # that has weights attributes with provided quantization config (in the tp model) to
                                 # its framework-specific attribute name.
                                 # note that a DefaultDict should be provided if not all the layer types in the
                                 # operation set are provided separately in the mapping.
                                 attr_mapping={
                                     KERNEL_ATTR: DefaultDict({
                                         DepthwiseConv2D: KERAS_DEPTHWISE_KERNEL,
                                         tf.nn.depthwise_conv2d: KERAS_DEPTHWISE_KERNEL}, default_value=KERAS_KERNEL),
                                     BIAS_ATTR: DefaultDict(default_value=BIAS)})
        tp.OperationsSetToLayers("FullyConnected", [Dense],
                                 attr_mapping={KERNEL_ATTR: DefaultDict(default_value=KERAS_KERNEL),
                                               BIAS_ATTR: DefaultDict(default_value=BIAS)})
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
        tp.OperationsSetToLayers("Div", [tf.math.divide, tf.math.truediv])
        tp.OperationsSetToLayers("PReLU", [PReLU])
        tp.OperationsSetToLayers("Swish", [tf.nn.swish, tp.LayerFilterParams(Activation, activation="swish")])
        tp.OperationsSetToLayers("Sigmoid", [tf.nn.sigmoid, tp.LayerFilterParams(Activation, activation="sigmoid")])
        tp.OperationsSetToLayers("Tanh", [tf.nn.tanh, tp.LayerFilterParams(Activation, activation="tanh")])

    return keras_tpc
