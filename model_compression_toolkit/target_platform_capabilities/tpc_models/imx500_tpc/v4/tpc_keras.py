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
from model_compression_toolkit.verify_packages import FOUND_SONY_CUSTOM_LAYERS
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, KERAS_DEPTHWISE_KERNEL, \
    KERAS_KERNEL, BIAS_ATTR, BIAS

if FOUND_SONY_CUSTOM_LAYERS:
    from sony_custom_layers.keras.object_detection.ssd_post_process import SSDPostProcess

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers import Conv2D, DepthwiseConv2D, Dense, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, Activation, ReLU, Add, Subtract, Multiply, PReLU, Flatten, Cropping2D, LeakyReLU, Permute, \
        Conv2DTranspose, Identity, Concatenate
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, Activation, ReLU, Add, Subtract, Multiply, PReLU, Flatten, Cropping2D, LeakyReLU, Permute, \
        Conv2DTranspose, Identity, Concatenate

from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v4.tp_model import get_tp_model
import model_compression_toolkit as mct
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v4 import __version__ as TPC_VERSION
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v4.tp_model import OPSET_NO_QUANTIZATION, \
    OPSET_QUANTIZATION_PRESERVING, OPSET_DIMENSION_MANIPULATION_OPS_WITH_WEIGHTS, OPSET_DIMENSION_MANIPULATION_OPS, \
    OPSET_MERGE_OPS, OPSET_CONV, OPSET_FULLY_CONNECTED, OPSET_ANY_RELU, OPSET_ADD, OPSET_SUB, OPSET_MUL, OPSET_DIV, \
    OPSET_PRELU, OPSET_SWISH, OPSET_SIGMOID, OPSET_TANH

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

    no_quant_list = [tf.quantization.fake_quant_with_min_max_vars,
                     tf.math.argmax,
                     tf.shape,
                     tf.math.equal,
                     tf.nn.top_k,
                     tf.image.combined_non_max_suppression,
                     tf.compat.v1.shape]
    quantization_preserving = [Cropping2D,
                               ZeroPadding2D,
                               Dropout,
                               MaxPooling2D,
                               tf.split,
                               tf.cast,
                               tf.unstack,
                               tf.__operators__.getitem,
                               tf.strided_slice]
    quantization_preserving_list_16bit_input = [Reshape,
                                                tf.reshape,
                                                Permute,
                                                tf.transpose,
                                                Flatten]

    if FOUND_SONY_CUSTOM_LAYERS:
        no_quant_list.append(SSDPostProcess)

    with keras_tpc:
        tp.OperationsSetToLayers(OPSET_NO_QUANTIZATION, no_quant_list)
        tp.OperationsSetToLayers(OPSET_QUANTIZATION_PRESERVING, quantization_preserving)
        tp.OperationsSetToLayers(OPSET_DIMENSION_MANIPULATION_OPS, quantization_preserving_list_16bit_input)
        tp.OperationsSetToLayers(OPSET_DIMENSION_MANIPULATION_OPS_WITH_WEIGHTS, [tf.gather, tf.compat.v1.gather])
        tp.OperationsSetToLayers(OPSET_MERGE_OPS, [tf.stack, tf.concat, Concatenate])
        tp.OperationsSetToLayers(OPSET_CONV,
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
        tp.OperationsSetToLayers(OPSET_FULLY_CONNECTED, [Dense],
                                 attr_mapping={KERNEL_ATTR: DefaultDict(default_value=KERAS_KERNEL),
                                               BIAS_ATTR: DefaultDict(default_value=BIAS)})
        tp.OperationsSetToLayers(OPSET_ANY_RELU, [tf.nn.relu,
                                                  tf.nn.relu6,
                                                  tf.nn.leaky_relu,
                                                  ReLU,
                                                  LeakyReLU,
                                                  tp.LayerFilterParams(Activation, activation="relu"),
                                                  tp.LayerFilterParams(Activation, activation="leaky_relu")])
        tp.OperationsSetToLayers(OPSET_ADD, [tf.add, Add])
        tp.OperationsSetToLayers(OPSET_SUB, [tf.subtract, Subtract])
        tp.OperationsSetToLayers(OPSET_MUL, [tf.math.multiply, Multiply])
        tp.OperationsSetToLayers(OPSET_DIV, [tf.math.divide, tf.math.truediv])
        tp.OperationsSetToLayers(OPSET_PRELU, [PReLU])
        tp.OperationsSetToLayers(OPSET_SWISH, [tf.nn.swish, tp.LayerFilterParams(Activation, activation="swish")])
        tp.OperationsSetToLayers(OPSET_SIGMOID, [tf.nn.sigmoid, tp.LayerFilterParams(Activation, activation="sigmoid")])
        tp.OperationsSetToLayers(OPSET_TANH, [tf.nn.tanh, tp.LayerFilterParams(Activation, activation="tanh")])

    return keras_tpc
