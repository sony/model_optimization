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
from typing import List, Tuple

import model_compression_toolkit as mct
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.constants import FLOAT_BITWIDTH
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR, WEIGHTS_N_BITS, \
    WEIGHTS_QUANTIZATION_METHOD, IMX500_TP_MODEL
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformModel, \
    Signedness, \
    AttributeQuantizationConfig, OpQuantizationConfig

tp = mct.target_platform


def get_tp_model() -> TargetPlatformModel:
    """
    A method that generates a default target platform model, with base 8-bit quantization configuration and 8, 4, 2
    bits configuration list for mixed-precision quantization.
    NOTE: in order to generate a target platform model with different configurations but with the same Operators Sets
    (for tests, experiments, etc.), use this method implementation as a test-case, i.e., override the
    'get_op_quantization_configs' method and use its output to call 'generate_tp_model' with your configurations.
    This version enables metadata by default.

    Returns: A TargetPlatformModel object.

    """
    base_config, mixed_precision_cfg_list, default_config = get_op_quantization_configs()
    return generate_tp_model(default_config=default_config,
                             base_config=base_config,
                             mixed_precision_cfg_list=mixed_precision_cfg_list,
                             name='imx500_lut_tp_model')


def get_op_quantization_configs() -> \
        Tuple[OpQuantizationConfig, List[OpQuantizationConfig], OpQuantizationConfig]:
    """
    Creates a default configuration object for 8-bit quantization, to be used to set a default TargetPlatformModel.
    In addition, creates a default configuration objects list (with 8, 4 and 2 bit quantization) to be used as
    default configuration for mixed-precision quantization with non-uniform quantizer for 2 and 4 bit candidates.

    Returns: An OpQuantizationConfig config object and a list of OpQuantizationConfig objects.

    """

    # We define a default quantization config for all non-specified weights attributes.
    default_weight_attr_config = AttributeQuantizationConfig(
        weights_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        weights_n_bits=8,
        weights_per_channel_threshold=False,
        enable_weights_quantization=False,
        lut_values_bitwidth=None)

    # define a quantization config to quantize the kernel (for layers where there is a kernel attribute).
    kernel_base_config = AttributeQuantizationConfig(
        weights_quantization_method=tp.QuantizationMethod.SYMMETRIC,
        weights_n_bits=8,
        weights_per_channel_threshold=True,
        enable_weights_quantization=True,
        lut_values_bitwidth=None)

    # We define a quantization config to quantize the bias (for layers where there is a bias attribute).
    bias_config = AttributeQuantizationConfig(
        weights_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
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
        default_weight_attr_config=default_weight_attr_config,
        attr_weights_configs_mapping={KERNEL_ATTR: kernel_base_config, BIAS_ATTR: bias_config},
        activation_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        simd_size=32,
        signedness=Signedness.AUTO)

    # To quantize a model using mixed-precision, create
    # a list with more than one OpQuantizationConfig.
    # In this example, we quantize some operations' weights
    # using 2, 4 or 8 bits, and when using 2 or 4 bits, it's possible
    # to quantize the operations' activations using LUT.
    four_bits_lut = linear_eight_bits.clone_and_edit(
        attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: 4,
                                    WEIGHTS_QUANTIZATION_METHOD: tp.QuantizationMethod.LUT_SYM_QUANTIZER}},
        simd_size=linear_eight_bits.simd_size * 2)
    two_bits_lut = linear_eight_bits.clone_and_edit(
        attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: 2,
                                    WEIGHTS_QUANTIZATION_METHOD: tp.QuantizationMethod.LUT_SYM_QUANTIZER}},
        simd_size=linear_eight_bits.simd_size * 4)
    mixed_precision_cfg_list = [linear_eight_bits, four_bits_lut, two_bits_lut]

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
    default_configuration_options = schema.QuantizationConfigOptions(
        quantization_configurations=tuple([default_config]))

    # Create a QuantizationConfigOptions for quantizing constants in functional ops.
    # Constant configuration is similar to the default eight bit configuration except for PoT
    # quantization method for the constant.
    # Since the constants are not named attributes of the layer, we use the default_weight_attr_config to
    # define the desired quantization properties for them.
    const_config = default_config.clone_and_edit(
        default_weight_attr_config=default_config.default_weight_attr_config.clone_and_edit(
            enable_weights_quantization=True, weights_per_channel_threshold=True,
            weights_quantization_method=tp.QuantizationMethod.POWER_OF_TWO))
    const_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple([const_config]))

    # 16 bits inputs and outputs. Currently, only defined for consts since they are used in operators that
    # support 16 bit as input and output.
    const_config_input16 = const_config.clone_and_edit(
        supported_input_activation_n_bits=(8, 16))
    const_config_input16_output16 = const_config_input16.clone_and_edit(
        activation_n_bits=16, signedness=Signedness.SIGNED)
    const_configuration_options_inout16 = schema.QuantizationConfigOptions(quantization_configurations=
                                                                           tuple([const_config_input16_output16,
                                                                                  const_config_input16]),
                                                                           base_config=const_config_input16)

    # Create Mixed-Precision quantization configuration options from the given list of OpQuantizationConfig objects
    mixed_precision_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=
                                                                             tuple(mixed_precision_cfg_list),
                                                                             base_config=base_config)

    # Create an OperatorsSet to represent a set of operations.
    # Each OperatorsSet has a unique label.
    # If a quantization configuration options is passed, these options will
    # be used for operations that will be attached to this set's label.
    # Otherwise, it will be a configure-less set (used in fusing):
    operator_set = []
    fusing_patterns = []
    # May suit for operations like: Dropout, Reshape, etc.

    no_quantization_config = default_configuration_options.clone_and_edit(
        enable_activation_quantization=False,
        supported_input_activation_n_bits=(8, 16)).clone_and_edit_weight_attribute(enable_weights_quantization=False)

    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.UNSTACK, qc_options=no_quantization_config))
    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.DROPOUT, qc_options=no_quantization_config))
    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.FLATTEN, qc_options=no_quantization_config))
    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.SPLIT_CHUNK, qc_options=no_quantization_config))
    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.GET_ITEM, qc_options=no_quantization_config))
    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.RESHAPE, qc_options=no_quantization_config))
    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.UNSQUEEZE, qc_options=no_quantization_config))
    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.BATCH_NORM, qc_options=no_quantization_config))
    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.SIZE, qc_options=no_quantization_config))
    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.PERMUTE, qc_options=no_quantization_config))
    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.TRANSPOSE, qc_options=no_quantization_config))
    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.EQUAL, qc_options=no_quantization_config))
    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.ARGMAX, qc_options=no_quantization_config))
    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.GATHER, qc_options=no_quantization_config))
    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.TOPK, qc_options=no_quantization_config))
    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.SQUEEZE, qc_options=no_quantization_config))
    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.MAXPOOL, qc_options=no_quantization_config))
    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.PAD, qc_options=no_quantization_config))
    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.ZERO_PADDING2D, qc_options=no_quantization_config))
    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.CAST, qc_options=no_quantization_config))
    operator_set.append(schema.OperatorsSet(name=schema.OperatorSetNames.COMBINED_NON_MAX_SUPPRESSION,
                                            qc_options=no_quantization_config))
    operator_set.append(schema.OperatorsSet(name=schema.OperatorSetNames.FAKE_QUANT,
                                            qc_options=no_quantization_config))
    operator_set.append(
        schema.OperatorsSet(name=schema.OperatorSetNames.STRIDED_SLICE, qc_options=no_quantization_config))
    operator_set.append(schema.OperatorsSet(name=schema.OperatorSetNames.SSD_POST_PROCESS,
                                            qc_options=no_quantization_config))

    operator_set.append(schema.OperatorsSet(name=schema.OperatorSetNames.STACK,
                                            qc_options=const_configuration_options_inout16))
    operator_set.append(schema.OperatorsSet(name=schema.OperatorSetNames.CONCATENATE,
                                            qc_options=const_configuration_options_inout16))

    # Define operator sets that use mixed_precision_configuration_options:
    conv = schema.OperatorsSet(name=schema.OperatorSetNames.CONV,
                               qc_options=mixed_precision_configuration_options)
    conv_transpose = schema.OperatorsSet(name=schema.OperatorSetNames.CONV_TRANSPOSE,
                                         qc_options=mixed_precision_configuration_options)
    depthwise_conv = schema.OperatorsSet(name=schema.OperatorSetNames.DEPTHWISE_CONV,
                                         qc_options=mixed_precision_configuration_options)
    fc = schema.OperatorsSet(name=schema.OperatorSetNames.FULLY_CONNECTED,
                             qc_options=mixed_precision_configuration_options)

    # Define operations sets without quantization configuration options (useful for creating fusing patterns):
    relu = schema.OperatorsSet(name=schema.OperatorSetNames.RELU)
    relu6 = schema.OperatorsSet(name=schema.OperatorSetNames.RELU6)
    leaky_relu = schema.OperatorsSet(name=schema.OperatorSetNames.LEAKY_RELU)
    prelu = schema.OperatorsSet(name=schema.OperatorSetNames.PRELU)
    add = schema.OperatorsSet(name=schema.OperatorSetNames.ADD,
                              qc_options=const_configuration_options_inout16)
    sub = schema.OperatorsSet(name=schema.OperatorSetNames.SUB,
                              qc_options=const_configuration_options_inout16)
    mul = schema.OperatorsSet(name=schema.OperatorSetNames.MUL,
                              qc_options=const_configuration_options_inout16)
    div = schema.OperatorsSet(name=schema.OperatorSetNames.DIV, qc_options=const_configuration_options)
    swish = schema.OperatorsSet(name=schema.OperatorSetNames.SWISH)
    hard_swish = schema.OperatorsSet(name=schema.OperatorSetNames.HARDSWISH)
    sigmoid = schema.OperatorsSet(name=schema.OperatorSetNames.SIGMOID)
    tanh = schema.OperatorsSet(name=schema.OperatorSetNames.TANH)
    hard_tanh = schema.OperatorsSet(name=schema.OperatorSetNames.HARD_TANH)

    operator_set.extend(
        [conv, conv_transpose, depthwise_conv, fc, relu, relu6, leaky_relu, add, sub, mul, div, prelu, swish,
         hard_swish, sigmoid,
         tanh, hard_tanh])
    any_relu = schema.OperatorSetConcat(operators_set=[relu, relu6, leaky_relu, hard_tanh])

    # Combine multiple operators into a single operator to avoid quantization between
    # them. To do this, we define fusing patterns using the OperatorsSets that were created.
    # To group multiple sets with regard to fusing, an OperatorSetConcat can be created
    activations_after_conv_to_fuse = schema.OperatorSetConcat(
        operators_set=[relu, relu6, leaky_relu, hard_tanh, swish, hard_swish, prelu, sigmoid, tanh])
    conv_types = schema.OperatorSetConcat(operators_set=[conv, conv_transpose, depthwise_conv])
    activations_after_fc_to_fuse = schema.OperatorSetConcat(
        operators_set=[relu, relu6, leaky_relu, hard_tanh, swish, hard_swish, sigmoid])
    any_binary = schema.OperatorSetConcat(operators_set=[add, sub, mul, div])

    # ------------------- #
    # Fusions
    # ------------------- #
    fusing_patterns.append(schema.Fusing(operator_groups=(conv_types, activations_after_conv_to_fuse)))
    fusing_patterns.append(schema.Fusing(operator_groups=(fc, activations_after_fc_to_fuse)))
    fusing_patterns.append(schema.Fusing(operator_groups=(any_binary, any_relu)))

    # Create a TargetPlatformModel and set its default quantization config.
    # This default configuration will be used for all operations
    # unless specified otherwise (see OperatorsSet, for example):
    generated_tpm = schema.TargetPlatformModel(
        default_qco=default_configuration_options,
        tpc_minor_version=3,
        tpc_patch_version=0,
        tpc_platform_type=IMX500_TP_MODEL,
        operator_set=tuple(operator_set),
        fusing_patterns=tuple(fusing_patterns),
        add_metadata=True,
        name=name,
        is_simd_padding=True)

    return generated_tpm