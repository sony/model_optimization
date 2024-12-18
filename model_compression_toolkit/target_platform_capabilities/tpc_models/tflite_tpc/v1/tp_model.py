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
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.constants import FLOAT_BITWIDTH
from model_compression_toolkit.target_platform_capabilities.constants import BIAS_ATTR, KERNEL_ATTR, TFLITE_TP_MODEL
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformModel, Signedness, \
    AttributeQuantizationConfig, OpQuantizationConfig

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
    base_config, mixed_precision_cfg_list, default_config = get_op_quantization_configs()
    return generate_tp_model(default_config=default_config,
                             base_config=base_config,
                             mixed_precision_cfg_list=mixed_precision_cfg_list,
                             name='tflite_tp_model')


def get_op_quantization_configs() -> Tuple[OpQuantizationConfig, List[OpQuantizationConfig], OpQuantizationConfig]:
    """
    Creates a default configuration object for 8-bit quantization, to be used to set a default TargetPlatformModel.
    In addition, creates a default configuration objects list (with 8, 4 and 2 bit quantization) to be used as
    default configuration for mixed-precision quantization.

    Returns: An OpQuantizationConfig config object and a list of OpQuantizationConfig objects.

    """

    # We define a default quantization config for all non-specified weights attributes.
    default_weight_attr_config = AttributeQuantizationConfig(
        weights_quantization_method=tp.QuantizationMethod.SYMMETRIC,
        weights_n_bits=8,
        weights_per_channel_threshold=False,
        enable_weights_quantization=False,
        lut_values_bitwidth=None)

    # We define a quantization config to quantize the kernel (for layers where there is a kernel attribute).
    kernel_base_config = AttributeQuantizationConfig(
        weights_quantization_method=tp.QuantizationMethod.SYMMETRIC,
        weights_n_bits=8,
        weights_per_channel_threshold=True,
        enable_weights_quantization=True,
        lut_values_bitwidth=None)

    # We define a quantization config to quantize the bias (for layers where there is a bias attribute).
    bias_config = AttributeQuantizationConfig(
        weights_quantization_method=tp.QuantizationMethod.SYMMETRIC,
        weights_n_bits=FLOAT_BITWIDTH,
        weights_per_channel_threshold=False,
        enable_weights_quantization=False,
        lut_values_bitwidth=None)

    # Create a quantization config.
    # A quantization configuration defines how an operator
    # should be quantized on the modeled hardware:

    # We define a default config for operation without kernel attribute.
    # This is the default config that should be used for non-linear operations.
    eight_bits_default = schema.OpQuantizationConfig(
        default_weight_attr_config=default_weight_attr_config,
        attr_weights_configs_mapping={},
        activation_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        simd_size=32,
        signedness=Signedness.AUTO)

    # We define an 8-bit config for linear operations quantization, that include a kernel and bias attributes.
    linear_eight_bits = schema.OpQuantizationConfig(
        activation_quantization_method=tp.QuantizationMethod.UNIFORM,
        default_weight_attr_config=default_weight_attr_config,
        attr_weights_configs_mapping={KERNEL_ATTR: kernel_base_config, BIAS_ATTR: bias_config},
        activation_n_bits=8,
        supported_input_activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        simd_size=None,
        signedness=Signedness.AUTO)

    mixed_precision_cfg_list = []  # No mixed precision

    return linear_eight_bits, mixed_precision_cfg_list, eight_bits_default


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
    default_configuration_options = schema.QuantizationConfigOptions(tuple([default_config]))

    # In TFLite, the quantized operator specifications constraint operators quantization
    # differently. For more details:
    # https://www.tensorflow.org/lite/performance/quantization_spec#int8_quantized_operator_specifications
    operator_set = []
    fusing_patterns = []

    operator_set.append(schema.OperatorsSet("NoQuantization",
                           default_configuration_options.clone_and_edit(
                               quantization_preserving=True)))

    fc = schema.OperatorsSet("FullyConnected",
                                default_configuration_options.clone_and_edit_weight_attribute(weights_per_channel_threshold=False))

    operator_set.append(schema.OperatorsSet("L2Normalization",
                           default_configuration_options.clone_and_edit(
                               fixed_zero_point=0, fixed_scale=1 / 128)))
    operator_set.append(schema.OperatorsSet("LogSoftmax",
                           default_configuration_options.clone_and_edit(
                               fixed_zero_point=127, fixed_scale=16 / 256)))
    operator_set.append(schema.OperatorsSet("Tanh",
                           default_configuration_options.clone_and_edit(
                               fixed_zero_point=0, fixed_scale=1 / 128)))
    operator_set.append(schema.OperatorsSet("Softmax",
                           default_configuration_options.clone_and_edit(
                               fixed_zero_point=-128, fixed_scale=1 / 256)))
    operator_set.append(schema.OperatorsSet("Logistic",
                           default_configuration_options.clone_and_edit(
                               fixed_zero_point=-128, fixed_scale=1 / 256)))

    conv2d = schema.OperatorsSet("Conv2d")
    kernel = schema.OperatorSetConcat([conv2d, fc])

    relu = schema.OperatorsSet("Relu")
    elu = schema.OperatorsSet("Elu")
    activations_to_fuse = schema.OperatorSetConcat([relu, elu])

    batch_norm = schema.OperatorsSet("BatchNorm")
    bias_add = schema.OperatorsSet("BiasAdd")
    add = schema.OperatorsSet("Add")
    squeeze = schema.OperatorsSet("Squeeze",
                                     qc_options=default_configuration_options.clone_and_edit(
                                         quantization_preserving=True))
    operator_set.extend([fc, conv2d, kernel, relu, elu, batch_norm, bias_add, add, squeeze])
    # ------------------- #
    # Fusions
    # ------------------- #
    # Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/optimizers/remapper
    fusing_patterns.append(schema.Fusing((kernel, bias_add)))
    fusing_patterns.append(schema.Fusing((kernel, bias_add, activations_to_fuse)))
    fusing_patterns.append(schema.Fusing((conv2d, batch_norm, activations_to_fuse)))
    fusing_patterns.append(schema.Fusing((conv2d, squeeze, activations_to_fuse)))
    fusing_patterns.append(schema.Fusing((batch_norm, activations_to_fuse)))
    fusing_patterns.append(schema.Fusing((batch_norm, add, activations_to_fuse)))

    # Create a TargetPlatformModel and set its default quantization config.
    # This default configuration will be used for all operations
    # unless specified otherwise (see OperatorsSet, for example):
    generated_tpc = schema.TargetPlatformModel(
        default_configuration_options,
        tpc_minor_version=1,
        tpc_patch_version=0,
        operator_set=tuple(operator_set),
        fusing_patterns=tuple(fusing_patterns),
        tpc_platform_type=TFLITE_TP_MODEL,
        add_metadata=False,
        name=name)

    return generated_tpc
