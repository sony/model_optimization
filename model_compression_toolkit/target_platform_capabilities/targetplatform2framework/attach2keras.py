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

from model_compression_toolkit.target_platform_capabilities.targetplatform2framework import LayerFilterParams
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2fw import \
    AttachTpcToFramework

from sony_custom_layers.keras.object_detection.ssd_post_process import SSDPostProcess

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers import Conv2D, DepthwiseConv2D, Dense, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, AveragePooling2D, Activation, ReLU, Add, Subtract, Multiply, PReLU, Flatten, Cropping2D, LeakyReLU, Permute, \
        Conv2DTranspose, Concatenate, BatchNormalization, Minimum, Maximum, Softmax
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, AveragePooling2D, Activation, ReLU, Add, Subtract, Multiply, PReLU, Flatten, Cropping2D, LeakyReLU, Permute, \
        Conv2DTranspose, Concatenate, BatchNormalization, Minimum, Maximum, Softmax

from model_compression_toolkit import DefaultDict
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS, \
    BIAS_ATTR, KERAS_KERNEL, KERAS_DEPTHWISE_KERNEL
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OperatorSetNames


class AttachTpcToKeras(AttachTpcToFramework):
    def __init__(self):
        super().__init__()

        self._opset2layer = {
            OperatorSetNames.CONV: [Conv2D, tf.nn.conv2d],
            OperatorSetNames.DEPTHWISE_CONV: [DepthwiseConv2D, tf.nn.depthwise_conv2d],
            OperatorSetNames.CONV_TRANSPOSE: [Conv2DTranspose, tf.nn.conv2d_transpose],
            OperatorSetNames.FULLY_CONNECTED: [Dense],
            OperatorSetNames.CONCATENATE: [tf.concat, Concatenate],
            OperatorSetNames.STACK: [tf.stack],
            OperatorSetNames.UNSTACK: [tf.unstack],
            OperatorSetNames.GATHER: [tf.gather, tf.compat.v1.gather],
            OperatorSetNames.EXPAND: [],
            OperatorSetNames.BATCH_NORM: [BatchNormalization, tf.nn.batch_normalization],
            OperatorSetNames.RELU: [tf.nn.relu, ReLU, LayerFilterParams(Activation, activation="relu")],
            OperatorSetNames.RELU6: [tf.nn.relu6],
            OperatorSetNames.LEAKY_RELU: [tf.nn.leaky_relu, LeakyReLU, LayerFilterParams(Activation, activation="leaky_relu")],
            OperatorSetNames.HARD_TANH: [LayerFilterParams(Activation, activation="hard_tanh")],
            OperatorSetNames.ADD: [tf.add, Add],
            OperatorSetNames.SUB: [tf.subtract, Subtract],
            OperatorSetNames.MUL: [tf.math.multiply, Multiply],
            OperatorSetNames.DIV: [tf.math.divide, tf.math.truediv],
            OperatorSetNames.MIN: [tf.math.minimum, Minimum],
            OperatorSetNames.MAX: [tf.math.maximum, Maximum],
            OperatorSetNames.PRELU: [PReLU],
            OperatorSetNames.SWISH: [tf.nn.swish, LayerFilterParams(Activation, activation="swish")],
            OperatorSetNames.HARDSWISH: [LayerFilterParams(Activation, activation="hard_swish")],
            OperatorSetNames.SIGMOID: [tf.nn.sigmoid, LayerFilterParams(Activation, activation="sigmoid")],
            OperatorSetNames.TANH: [tf.nn.tanh, LayerFilterParams(Activation, activation="tanh")],
            OperatorSetNames.GELU: [tf.nn.gelu, LayerFilterParams(Activation, activation="gelu")],
            OperatorSetNames.HARDSIGMOID: [tf.keras.activations.hard_sigmoid,
                                           LayerFilterParams(Activation, activation="hard_sigmoid")],
            OperatorSetNames.FLATTEN: [Flatten],
            OperatorSetNames.GET_ITEM: [tf.__operators__.getitem],
            OperatorSetNames.RESHAPE: [Reshape, tf.reshape],
            OperatorSetNames.PERMUTE: [Permute],
            OperatorSetNames.TRANSPOSE: [tf.transpose],
            OperatorSetNames.UNSQUEEZE: [tf.expand_dims],
            OperatorSetNames.SQUEEZE: [tf.squeeze],
            OperatorSetNames.DROPOUT: [Dropout],
            OperatorSetNames.SPLIT_CHUNK: [tf.split],
            OperatorSetNames.MAXPOOL: [MaxPooling2D, tf.nn.avg_pool2d],
            OperatorSetNames.AVGPOOL: [AveragePooling2D],
            OperatorSetNames.SIZE: [tf.size],
            OperatorSetNames.RESIZE: [tf.image.resize],
            OperatorSetNames.PAD: [tf.pad, Cropping2D],
            OperatorSetNames.FOLD: [tf.space_to_batch_nd],
            OperatorSetNames.SHAPE: [tf.shape, tf.compat.v1.shape],
            OperatorSetNames.EQUAL: [tf.math.equal],
            OperatorSetNames.ARGMAX: [tf.math.argmax],
            OperatorSetNames.TOPK: [tf.nn.top_k],
            OperatorSetNames.FAKE_QUANT: [tf.quantization.fake_quant_with_min_max_vars],
            OperatorSetNames.COMBINED_NON_MAX_SUPPRESSION: [tf.image.combined_non_max_suppression],
            OperatorSetNames.ZERO_PADDING2D: [ZeroPadding2D],
            OperatorSetNames.CAST: [tf.cast],
            OperatorSetNames.STRIDED_SLICE: [tf.strided_slice],
            OperatorSetNames.ELU: [tf.nn.elu, LayerFilterParams(Activation, activation="elu")],
            OperatorSetNames.SOFTMAX: [tf.nn.softmax, Softmax,
                                       LayerFilterParams(Activation, activation="softmax")],
            OperatorSetNames.LOG_SOFTMAX: [tf.nn.log_softmax],
            OperatorSetNames.ADD_BIAS: [tf.nn.bias_add],
            OperatorSetNames.L2NORM: [tf.math.l2_normalize],
            OperatorSetNames.SSD_POST_PROCESS: [SSDPostProcess]
        }

        self._opset2attr_mapping = {
            OperatorSetNames.CONV: {
                KERNEL_ATTR: DefaultDict(default_value=KERAS_KERNEL),
                BIAS_ATTR: DefaultDict(default_value=BIAS)},
            OperatorSetNames.CONV_TRANSPOSE: {
                KERNEL_ATTR: DefaultDict(default_value=KERAS_KERNEL),
                BIAS_ATTR: DefaultDict(default_value=BIAS)},
            OperatorSetNames.DEPTHWISE_CONV: {
                KERNEL_ATTR: DefaultDict({
                    DepthwiseConv2D: KERAS_DEPTHWISE_KERNEL,
                    tf.nn.depthwise_conv2d: KERAS_DEPTHWISE_KERNEL}, default_value=KERAS_KERNEL),
                BIAS_ATTR: DefaultDict(default_value=BIAS)},
            OperatorSetNames.FULLY_CONNECTED: {
                KERNEL_ATTR: DefaultDict(default_value=KERAS_KERNEL),
                BIAS_ATTR: DefaultDict(default_value=BIAS)}}
