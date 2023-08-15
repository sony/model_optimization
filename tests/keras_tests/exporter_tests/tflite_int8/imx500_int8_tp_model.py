# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import List, Tuple

import tensorflow as tf
from packaging import version

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers import Conv2D, DepthwiseConv2D, Dense, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, Activation, ReLU, Add, Subtract, Multiply, PReLU, Flatten, Cropping2D, LeakyReLU, Permute, \
        Conv2DTranspose
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, Activation, ReLU, Add, Subtract, Multiply, PReLU, Flatten, Cropping2D, LeakyReLU, Permute, \
        Conv2DTranspose

import model_compression_toolkit as mct
from model_compression_toolkit.target_platform_capabilities.target_platform import OpQuantizationConfig, \
    TargetPlatformModel
from model_compression_toolkit.target_platform_capabilities.target_platform.quantization_format import \
    QuantizationFormat

tp = mct.target_platform


def get_tp_model(edit_params_dict) -> TargetPlatformModel:
    base_config, mixed_precision_cfg_list = get_op_quantization_configs()

    updated_config = base_config.clone_and_edit(**edit_params_dict)
    op_cfg_list = [updated_config]

    return generate_tp_model(default_config=updated_config,
                             base_config=updated_config,
                             mixed_precision_cfg_list=op_cfg_list,
                             name='int8_tp_model')


def get_op_quantization_configs() -> Tuple[OpQuantizationConfig, List[OpQuantizationConfig]]:
    eight_bits = tp.OpQuantizationConfig(
        activation_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        weights_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        weights_n_bits=8,
        weights_per_channel_threshold=True,
        enable_weights_quantization=True,
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        weights_multiplier_nbits=None)
    four_bits = eight_bits.clone_and_edit(weights_n_bits=4)
    two_bits = eight_bits.clone_and_edit(weights_n_bits=2)
    mixed_precision_cfg_list = [eight_bits, four_bits, two_bits]
    return eight_bits, mixed_precision_cfg_list


def generate_tp_model(default_config: OpQuantizationConfig,
                      base_config: OpQuantizationConfig,
                      mixed_precision_cfg_list: List[OpQuantizationConfig],
                      name: str) -> TargetPlatformModel:
    default_configuration_options = tp.QuantizationConfigOptions([default_config])
    generated_tpc = tp.TargetPlatformModel(default_configuration_options, name=name)
    with generated_tpc:
        generated_tpc.set_quantization_format(QuantizationFormat.INT8)
        tp.OperatorsSet("NoQuantization",
                        tp.get_default_quantization_config_options().clone_and_edit(
                            enable_weights_quantization=False,
                            enable_activation_quantization=False))
        mixed_precision_configuration_options = tp.QuantizationConfigOptions(mixed_precision_cfg_list,
                                                                             base_config=base_config)

        conv = tp.OperatorsSet("Conv", mixed_precision_configuration_options)
        fc = tp.OperatorsSet("FullyConnected", mixed_precision_configuration_options)

        any_relu = tp.OperatorsSet("AnyReLU")
        add = tp.OperatorsSet("Add")
        sub = tp.OperatorsSet("Sub")
        mul = tp.OperatorsSet("Mul")
        div = tp.OperatorsSet("Div")
        prelu = tp.OperatorsSet("PReLU")
        swish = tp.OperatorsSet("Swish")
        sigmoid = tp.OperatorsSet("Sigmoid")
        tanh = tp.OperatorsSet("Tanh")
        activations_after_conv_to_fuse = tp.OperatorSetConcat(any_relu, swish, prelu, sigmoid, tanh)
        activations_after_fc_to_fuse = tp.OperatorSetConcat(any_relu, swish, sigmoid)
        any_binary = tp.OperatorSetConcat(add, sub, mul, div)
        tp.Fusing([conv, activations_after_conv_to_fuse])
        tp.Fusing([fc, activations_after_fc_to_fuse])
        tp.Fusing([any_binary, any_relu])

    return generated_tpc


def get_int8_tpc(edit_params_dict) -> tp.TargetPlatformCapabilities:
    default_tp_model = get_tp_model(edit_params_dict)
    return generate_keras_tpc(name='int8_tpc', tp_model=default_tp_model)


def generate_keras_tpc(name: str, tp_model: tp.TargetPlatformModel):
    keras_tpc = tp.TargetPlatformCapabilities(tp_model, name=name, version='v1')

    with keras_tpc:
        tp.OperationsSetToLayers("NoQuantization", [Reshape,
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
                                                    tf.compat.v1.gather,
                                                    tf.nn.top_k,
                                                    tf.__operators__.getitem,
                                                    tf.compat.v1.shape])

        tp.OperationsSetToLayers("Conv", [Conv2D,
                                          DepthwiseConv2D,
                                          Conv2DTranspose,
                                          tf.nn.conv2d,
                                          tf.nn.depthwise_conv2d,
                                          tf.nn.conv2d_transpose])
        tp.OperationsSetToLayers("FullyConnected", [Dense])
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
        tp.OperationsSetToLayers("Div", [tf.math.divide])
        tp.OperationsSetToLayers("PReLU", [PReLU])
        tp.OperationsSetToLayers("Swish", [tf.nn.swish, tp.LayerFilterParams(Activation, activation="swish")])
        tp.OperationsSetToLayers("Sigmoid", [tf.nn.sigmoid, tp.LayerFilterParams(Activation, activation="sigmoid")])
        tp.OperationsSetToLayers("Tanh", [tf.nn.tanh, tp.LayerFilterParams(Activation, activation="tanh")])
    return keras_tpc
