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
from typing import List, Tuple

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.target_platform import OpQuantizationConfig, TargetPlatformModel
from model_compression_toolkit.core.common.target_platform.op_quantization_config import QuantizationMethod

tp = mct.target_platform


def get_tp_model() -> TargetPlatformModel:
    """
    A method that generates a default target platform model, with base 8-bit quantization configuration and 8, 4, 2
    bits configuration list for mixed-precision quantization.
    NOTE: in order to generate a target platform model with different configurations but with the same Operators Sets
    (for tests, experiments, etc.), use this method implementation as a test-case, i.e., override the
    'get_op_quantization_configs' method and use its output to call 'generate_tp_model' with your configurations.

    Returns: A TargetPlatformModel object.

    """
    base_config, mixed_precision_cfg_list = get_op_quantization_configs()
    return generate_tp_model(default_config=base_config,
                             base_config=base_config,
                             mixed_precision_cfg_list=mixed_precision_cfg_list,
                             name='tflite_tp_model')


def get_op_quantization_configs() -> Tuple[OpQuantizationConfig, List[OpQuantizationConfig]]:
    """
    Creates a default configuration object for 8-bit quantization, to be used to set a default TargetPlatformModel.
    In addition, creates a default configuration objects list (with 8, 4 and 2 bit quantization) to be used as
    default configuration for mixed-precision quantization.

    Returns: An OpQuantizationConfig config object and a list of OpQuantizationConfig objects.

    """
    # Create a quantization config.
    # A quantization configuration defines how an operator
    # should be quantized on the modeled hardware:
    eight_bits = tp.OpQuantizationConfig(
        activation_quantization_method=QuantizationMethod.UNIFORM,
        weights_quantization_method=QuantizationMethod.SYMMETRIC,
        activation_n_bits=8,
        weights_n_bits=8,
        weights_per_channel_threshold=True,
        enable_weights_quantization=True,
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        weights_multiplier_nbits=None
    )

    mixed_precision_cfg_list = [] # No mixed precision

    return eight_bits, mixed_precision_cfg_list


def generate_tp_model(default_config: OpQuantizationConfig,
                      base_config: OpQuantizationConfig,
                      mixed_precision_cfg_list: List[OpQuantizationConfig],
                      name: str) -> TargetPlatformModel:
    """
    Generates TargetPlatformModel with default defined Operators Sets, based on the given base configuration and
    mixed-precision configurations options list.

    Args
        default_config: A default OpQuantizationConfig to set as the TP model default configuration.
        base_config: An OpQuantizationConfig to set as the TargetPlatformModel base configuration for mixed-precision purposes only.
        mixed_precision_cfg_list: A list of OpQuantizationConfig to be used as the TP model mixed-precision
            quantization configuration options.
        name: The name of the TargetPlatformModel.

    Returns: A TargetPlatformModel object.

    """
    # Create a QuantizationConfigOptions, which defines a set
    # of possible configurations to consider when quantizing a set of operations (in mixed-precision, for example).
    # If the QuantizationConfigOptions contains only one configuration,
    # this configuration will be used for the operation quantization:
    default_configuration_options = tp.QuantizationConfigOptions([default_config])

    # Create a TargetPlatformModel and set its default quantization config.
    # This default configuration will be used for all operations
    # unless specified otherwise (see OperatorsSet, for example):
    generated_tpc = tp.TargetPlatformModel(default_configuration_options, name=name)

    # To start defining the model's components (such as operator sets, and fusing patterns),
    # use 'with' the TargetPlatformModel instance, and create them as below:
    with generated_tpc:
        # In TFLite, the quantized operator specifications constraint operators quantization
        # differently. For more details:
        # https://www.tensorflow.org/lite/performance/quantization_spec#int8_quantized_operator_specifications
        tp.OperatorsSet("NoQuantization",
                         tp.get_default_quantization_config_options().clone_and_edit(
                                  quantization_preserving=True))

        fc = tp.OperatorsSet("FullyConnected",
                              tp.get_default_quantization_config_options().clone_and_edit(
                                       weights_per_channel_threshold=False))

        tp.OperatorsSet("L2Normalization",
                         tp.get_default_quantization_config_options().clone_and_edit(
                                  fixed_zero_point=0, fixed_scale=1 / 128))
        tp.OperatorsSet("LogSoftmax",
                         tp.get_default_quantization_config_options().clone_and_edit(
                                  fixed_zero_point=127, fixed_scale=16 / 256))
        tp.OperatorsSet("Tanh",
                         tp.get_default_quantization_config_options().clone_and_edit(
                                  fixed_zero_point=0, fixed_scale=1 / 128))
        tp.OperatorsSet("Softmax",
                         tp.get_default_quantization_config_options().clone_and_edit(
                                  fixed_zero_point=-128, fixed_scale=1 / 256))
        tp.OperatorsSet("Logistic",
                         tp.get_default_quantization_config_options().clone_and_edit(
                                  fixed_zero_point=-128, fixed_scale=1 / 256))

        conv2d = tp.OperatorsSet("Conv2d")
        kernel = tp.OperatorSetConcat(conv2d, fc)

        relu = tp.OperatorsSet("Relu")
        elu = tp.OperatorsSet("Elu")
        activations_to_fuse = tp.OperatorSetConcat(relu, elu)

        batch_norm = tp.OperatorsSet("BatchNorm")
        bias_add = tp.OperatorsSet("BiasAdd")
        add = tp.OperatorsSet("Add")
        squeeze = tp.OperatorsSet("Squeeze",
                                   qc_options=tp.get_default_quantization_config_options().clone_and_edit(quantization_preserving=True))
        # ------------------- #
        # Fusions
        # ------------------- #
        # Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/optimizers/remapper
        tp.Fusing([kernel, bias_add])
        tp.Fusing([kernel, bias_add, activations_to_fuse])
        tp.Fusing([conv2d, batch_norm, activations_to_fuse])
        tp.Fusing([conv2d, squeeze, activations_to_fuse])
        tp.Fusing([batch_norm, activations_to_fuse])
        tp.Fusing([batch_norm, add, activations_to_fuse])

    return generated_tpc
