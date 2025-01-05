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
from model_compression_toolkit.target_platform_capabilities.target_platform import LayerFilterParams
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework.attach2fw import \
    AttachTpcToFramework


class AttachTpcToKeras(AttachTpcToFramework):
    def __init__(self):
        super().__init__()

        self._opset2layer = {
            OperatorSetNames.OPSET_CONV: [Conv2D, tf.nn.conv2d],
            OperatorSetNames.OPSET_DEPTHWISE_CONV: [DepthwiseConv2D, tf.nn.depthwise_conv2d],
            OperatorSetNames.OPSET_CONV_TRANSPOSE: [Conv2DTranspose, tf.nn.conv2d_transpose],
            OperatorSetNames.OPSET_FULLY_CONNECTED: [Dense],
            OperatorSetNames.OPSET_CONCATENATE: [tf.concat, Concatenate],
            OperatorSetNames.OPSET_STACK: [tf.stack],
            OperatorSetNames.OPSET_UNSTACK: [tf.unstack],
            OperatorSetNames.OPSET_GATHER: [tf.gather, tf.compat.v1.gather],
            OperatorSetNames.OPSET_EXPAND: [],
            OperatorSetNames.OPSET_BATCH_NORM: [BatchNormalization, tf.nn.batch_normalization],
            OperatorSetNames.OPSET_RELU: [tf.nn.relu, ReLU, LayerFilterParams(Activation, activation="relu")],
            OperatorSetNames.OPSET_RELU6: [tf.nn.relu6],
            OperatorSetNames.OPSET_LEAKY_RELU: [tf.nn.leaky_relu, LeakyReLU, LayerFilterParams(Activation, activation="leaky_relu")],
            OperatorSetNames.OPSET_HARD_TANH: [LayerFilterParams(Activation, activation="hard_tanh")],
            OperatorSetNames.OPSET_ADD: [tf.add, Add],
            OperatorSetNames.OPSET_SUB: [tf.subtract, Subtract],
            OperatorSetNames.OPSET_MUL: [tf.math.multiply, Multiply],
            OperatorSetNames.OPSET_DIV: [tf.math.divide, tf.math.truediv],
            OperatorSetNames.OPSET_MIN: [tf.math.minimum, Minimum],
            OperatorSetNames.OPSET_MAX: [tf.math.maximum, Maximum],
            OperatorSetNames.OPSET_PRELU: [PReLU],
            OperatorSetNames.OPSET_SWISH: [tf.nn.swish, LayerFilterParams(Activation, activation="swish")],
            OperatorSetNames.OPSET_HARDSWISH: [LayerFilterParams(Activation, activation="hard_swish")],
            OperatorSetNames.OPSET_SIGMOID: [tf.nn.sigmoid, LayerFilterParams(Activation, activation="sigmoid")],
            OperatorSetNames.OPSET_TANH: [tf.nn.tanh, LayerFilterParams(Activation, activation="tanh")],
            OperatorSetNames.OPSET_GELU: [tf.nn.gelu, LayerFilterParams(Activation, activation="gelu")],
            OperatorSetNames.OPSET_HARDSIGMOID: [tf.keras.activations.hard_sigmoid,
                                                       LayerFilterParams(Activation, activation="hard_sigmoid")],
            OperatorSetNames.OPSET_FLATTEN: [Flatten],
            OperatorSetNames.OPSET_GET_ITEM: [tf.__operators__.getitem],
            OperatorSetNames.OPSET_RESHAPE: [Reshape, tf.reshape],
            OperatorSetNames.OPSET_PERMUTE: [Permute],
            OperatorSetNames.OPSET_TRANSPOSE: [tf.transpose],
            OperatorSetNames.OPSET_UNSQUEEZE: [tf.expand_dims],
            OperatorSetNames.OPSET_SQUEEZE: [tf.squeeze],
            OperatorSetNames.OPSET_DROPOUT: [Dropout],
            OperatorSetNames.OPSET_SPLIT_CHUNK: [tf.split],
            OperatorSetNames.OPSET_MAXPOOL: [MaxPooling2D, tf.nn.avg_pool2d],
            OperatorSetNames.OPSET_AVGPOOL: [AveragePooling2D],
            OperatorSetNames.OPSET_SIZE: [tf.size],
            OperatorSetNames.OPSET_RESIZE: [tf.image.resize],
            OperatorSetNames.OPSET_PAD: [tf.pad, Cropping2D],
            OperatorSetNames.OPSET_FOLD: [tf.space_to_batch_nd],
            OperatorSetNames.OPSET_SHAPE: [tf.shape, tf.compat.v1.shape],
            OperatorSetNames.OPSET_EQUAL: [tf.math.equal],
            OperatorSetNames.OPSET_ARGMAX: [tf.math.argmax],
            OperatorSetNames.OPSET_TOPK: [tf.nn.top_k],
            OperatorSetNames.OPSET_FAKE_QUANT: [tf.quantization.fake_quant_with_min_max_vars],
            OperatorSetNames.OPSET_COMBINED_NON_MAX_SUPPRESSION: [tf.image.combined_non_max_suppression],
            OperatorSetNames.OPSET_ZERO_PADDING2d: [ZeroPadding2D],
            OperatorSetNames.OPSET_CAST: [tf.cast],
            OperatorSetNames.OPSET_STRIDED_SLICE: [tf.strided_slice],
            OperatorSetNames.OPSET_ELU: [tf.nn.elu, LayerFilterParams(Activation, activation="elu")],
            OperatorSetNames.OPSET_SOFTMAX: [tf.nn.softmax, Softmax,
                                                   LayerFilterParams(Activation, activation="softmax")],
            OperatorSetNames.OPSET_LOG_SOFTMAX: [tf.nn.log_softmax],
            OperatorSetNames.OPSET_ADD_BIAS: [tf.nn.bias_add],
            OperatorSetNames.OPSET_L2NORM: [tf.math.l2_normalize],
        }

        if FOUND_SONY_CUSTOM_LAYERS:
            self._opset2layer[OperatorSetNames.OPSET_SSD_POST_PROCESS] = [SSDPostProcess]
        else:
            # If Custom layers is not installed then we don't want the user to fail, but just ignore custom layers
            # in the initialized framework TPC
            self._opset2layer[OperatorSetNames.OPSET_SSD_POST_PROCESS] = []

        self._opset2attr_mapping = {
            OperatorSetNames.OPSET_CONV: {
                KERNEL_ATTR: DefaultDict(default_value=KERAS_KERNEL),
                BIAS_ATTR: DefaultDict(default_value=BIAS)},
            OperatorSetNames.OPSET_CONV_TRANSPOSE: {
                KERNEL_ATTR: DefaultDict(default_value=KERAS_KERNEL),
                BIAS_ATTR: DefaultDict(default_value=BIAS)},
            OperatorSetNames.OPSET_DEPTHWISE_CONV: {
                KERNEL_ATTR: DefaultDict({
                    DepthwiseConv2D: KERAS_DEPTHWISE_KERNEL,
                    tf.nn.depthwise_conv2d: KERAS_DEPTHWISE_KERNEL}, default_value=KERAS_KERNEL),
                BIAS_ATTR: DefaultDict(default_value=BIAS)},
            OperatorSetNames.OPSET_FULLY_CONNECTED: {
                KERNEL_ATTR: DefaultDict(default_value=KERAS_KERNEL),
                BIAS_ATTR: DefaultDict(default_value=BIAS)}}
