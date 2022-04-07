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
from keras.engine.input_layer import InputLayer

import model_compression_toolkit as mct
import tensorflow as tf

if tf.__version__ < "2.6":
    from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, Activation, ReLU, Add, PReLU, Flatten, Cropping2D
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose, Reshape, ZeroPadding2D, \
    Dropout, MaxPooling2D, Activation, ReLU, Add, PReLU, Flatten, Cropping2D


hwm = mct.hardware_representation


def generate_hw_model_with_activation_mp(base_cfg, mp_bitwidth_candidates_list, name="activation_mp_model"):

    # prepare mp candidates
    mixed_precision_cfg_list = []
    for weights_n_bits, activation_n_bits in mp_bitwidth_candidates_list:
        candidate_cfg = base_cfg.clone_and_edit(weights_n_bits=weights_n_bits,
                                                activation_n_bits=activation_n_bits)
        mixed_precision_cfg_list.append(candidate_cfg)

    # set hw model
    default_configuration_options = hwm.QuantizationConfigOptions([base_cfg])

    generated_hwm = hwm.HardwareModel(default_configuration_options, name=name)

    with generated_hwm:
        hwm.OperatorsSet("NoQuantization",
                         hwm.get_default_quantization_config_options().clone_and_edit(
                             enable_weights_quantization=False,
                             enable_activation_quantization=False))

        mixed_precision_configuration_options = hwm.QuantizationConfigOptions(mixed_precision_cfg_list,
                                                                              base_config=base_cfg)

        hwm.OperatorsSet("Weights_n_Activation", mixed_precision_configuration_options)
        hwm.OperatorsSet("Activation", mixed_precision_configuration_options)

    return generated_hwm


def generate_fhw_model_keras(hardware_model, name="activation_mp_keras_hwm"):

    fhwm_keras = hwm.FrameworkHardwareModel(hardware_model,
                                            name=name)
    with fhwm_keras:
        hwm.OperationsSetToLayers("NoQuantization", [Reshape,
                                                     tf.reshape,
                                                     Flatten,
                                                     Cropping2D,
                                                     ZeroPadding2D,
                                                     Dropout,
                                                     MaxPooling2D,
                                                     tf.split,
                                                     tf.quantization.fake_quant_with_min_max_vars])

        hwm.OperationsSetToLayers("Weights_n_Activation", [Conv2D,
                                                           DepthwiseConv2D,
                                                           tf.nn.conv2d,
                                                           tf.nn.depthwise_conv2d,
                                                           Dense,
                                                           Conv2DTranspose,
                                                           tf.nn.conv2d_transpose])

        hwm.OperationsSetToLayers("Activation", [tf.nn.relu,
                                                 tf.nn.relu6,
                                                 hwm.LayerFilterParams(ReLU, negative_slope=0.0),
                                                 hwm.LayerFilterParams(Activation, activation="relu"),
                                                 tf.add,
                                                 Add,
                                                 PReLU,
                                                 tf.nn.swish,
                                                 hwm.LayerFilterParams(Activation, activation="swish"),
                                                 tf.nn.sigmoid,
                                                 hwm.LayerFilterParams(Activation, activation="sigmoid"),
                                                 tf.nn.tanh,
                                                 hwm.LayerFilterParams(Activation, activation="tanh"),
                                                 InputLayer])

    return fhwm_keras