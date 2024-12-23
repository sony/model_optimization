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

from model_compression_toolkit.verify_packages import FOUND_SONY_CUSTOM_LAYERS

if FOUND_SONY_CUSTOM_LAYERS:
    from sony_custom_layers.keras.object_detection.ssd_post_process import SSDPostProcess

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers import Conv2D, DepthwiseConv2D, Dense, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, Activation, ReLU, Add, Subtract, Multiply, PReLU, Flatten, Cropping2D, LeakyReLU, Permute, \
        Conv2DTranspose, Identity, Concatenate, BatchNormalization, Minimum, Maximum
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, Activation, ReLU, Add, Subtract, Multiply, PReLU, Flatten, Cropping2D, LeakyReLU, Permute, \
        Conv2DTranspose, Concatenate, BatchNormalization, Minimum, Maximum

from model_compression_toolkit import DefaultDict
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS, \
    BIAS_ATTR, KERAS_KERNEL, KERAS_DEPTHWISE_KERNEL
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OperatorSetNames
from model_compression_toolkit.target_platform_capabilities.target_platform import LayerFilterParams
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework.attach2fw import \
    AttachTpModelToFw


class AttachTpModelToKeras(AttachTpModelToFw):
    def __init__(self):
        super().__init__()

        self._opset2layer = {
            OperatorSetNames.OPSET_CONV.value: [Conv2D, tf.nn.conv2d],
            OperatorSetNames.OPSET_DEPTHWISE_CONV.value: [DepthwiseConv2D, tf.nn.depthwise_conv2d],
            OperatorSetNames.OPSET_CONV_TRANSPOSE.value: [Conv2DTranspose, tf.nn.conv2d_transpose],
            OperatorSetNames.OPSET_FULLY_CONNECTED.value: [Dense],
            OperatorSetNames.OPSET_CONCATENATE.value: [tf.concat, Concatenate],
            OperatorSetNames.OPSET_STACK.value: [tf.stack],
            OperatorSetNames.OPSET_UNSTACK.value: [tf.unstack],
            OperatorSetNames.OPSET_GATHER.value: [tf.gather, tf.compat.v1.gather],
            OperatorSetNames.OPSET_EXPAND.value: [],
            OperatorSetNames.OPSET_BATCH_NORM.value: [BatchNormalization],
            OperatorSetNames.OPSET_RELU.value: [tf.nn.relu, ReLU],
            OperatorSetNames.OPSET_RELU6.value: [tf.nn.relu6],
            OperatorSetNames.OPSET_LEAKY_RELU.value: [tf.nn.leaky_relu, LeakyReLU],
            OperatorSetNames.OPSET_HARD_TANH.value: [LayerFilterParams(Activation, activation="hard_tanh")],
            OperatorSetNames.OPSET_ADD.value: [tf.add, Add],
            OperatorSetNames.OPSET_SUB.value: [tf.subtract, Subtract],
            OperatorSetNames.OPSET_MUL.value: [tf.math.multiply, Multiply],
            OperatorSetNames.OPSET_DIV.value: [tf.math.divide, tf.math.truediv],
            OperatorSetNames.OPSET_MIN.value: [tf.math.minimum, Minimum],
            OperatorSetNames.OPSET_MAX.value: [tf.math.maximum, Maximum],
            OperatorSetNames.OPSET_PRELU.value: [PReLU],
            OperatorSetNames.OPSET_SWISH.value: [tf.nn.swish, LayerFilterParams(Activation, activation="swish")],
            OperatorSetNames.OPSET_SIGMOID.value: [tf.nn.sigmoid, LayerFilterParams(Activation, activation="sigmoid")],
            OperatorSetNames.OPSET_TANH.value: [tf.nn.tanh, LayerFilterParams(Activation, activation="tanh")],
            OperatorSetNames.OPSET_GELU.value: [tf.nn.gelu, LayerFilterParams(Activation, activation="gelu")],
            OperatorSetNames.OPSET_HARDSIGMOID.value: [tf.keras.activations.hard_sigmoid,
                                                       LayerFilterParams(Activation, activation="hard_sigmoid")],
            OperatorSetNames.OPSET_FLATTEN.value: [Flatten],
            OperatorSetNames.OPSET_GET_ITEM.value: [tf.__operators__.getitem],
            OperatorSetNames.OPSET_RESHAPE.value: [Reshape, tf.reshape],
            OperatorSetNames.OPSET_PERMUTE.value: [Permute],
            OperatorSetNames.OPSET_TRANSPOSE.value: [tf.transpose],
            OperatorSetNames.OPSET_DROPOUT.value: [Dropout],
            OperatorSetNames.OPSET_SPLIT.value: [tf.split],
            OperatorSetNames.OPSET_MAXPOOL.value: [MaxPooling2D],
            OperatorSetNames.OPSET_SHAPE.value: [tf.shape, tf.compat.v1.shape],
            OperatorSetNames.OPSET_EQUAL.value: [tf.math.equal],
            OperatorSetNames.OPSET_ARGMAX.value: [tf.math.argmax],
            OperatorSetNames.OPSET_TOPK.value: [tf.nn.top_k],
            OperatorSetNames.OPSET_FAKE_QUANT_WITH_MIN_MAX_VARS.value: [tf.quantization.fake_quant_with_min_max_vars],
            OperatorSetNames.OPSET_COMBINED_NON_MAX_SUPPRESSION.value: [tf.image.combined_non_max_suppression],
            OperatorSetNames.OPSET_CROPPING2D.value: [Cropping2D],
            OperatorSetNames.OPSET_ZERO_PADDING2d.value: [ZeroPadding2D],
            OperatorSetNames.OPSET_CAST.value: [tf.cast],
            OperatorSetNames.OPSET_STRIDED_SLICE.value: [tf.strided_slice]
        }

        if FOUND_SONY_CUSTOM_LAYERS:
            self._opset2layer[OperatorSetNames.OPSET_POST_PROCESS] = [SSDPostProcess]

        self._opset2attr_mapping = {OperatorSetNames.OPSET_CONV.value: {
            KERNEL_ATTR: DefaultDict(default_value=KERAS_KERNEL),
            BIAS_ATTR: DefaultDict(default_value=BIAS)},
            OperatorSetNames.OPSET_DEPTHWISE_CONV.value: {
                KERNEL_ATTR: DefaultDict({
                    DepthwiseConv2D: KERAS_DEPTHWISE_KERNEL,
                    tf.nn.depthwise_conv2d: KERAS_DEPTHWISE_KERNEL}, default_value=KERAS_KERNEL),
                BIAS_ATTR: DefaultDict(default_value=BIAS)},
            OperatorSetNames.OPSET_FULLY_CONNECTED.value: {
                KERNEL_ATTR: DefaultDict(default_value=KERAS_KERNEL),
                BIAS_ATTR: DefaultDict(default_value=BIAS)}}
