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
from typing import List, Tuple

import model_compression_toolkit as mct
from model_compression_toolkit.common.hardware_representation import OpQuantizationConfig, HardwareModel

hwm = mct.hardware_representation


def get_default_hardware_model() -> HardwareModel:
    """
    A method that generates a default hardware model, with base 8-bit quantization configuration and 8, 4, 2
    bits configuration list for mixed-precision quantization.
    NOTE: in order to generate a hardware model with different configurations but with the same Operators Sets
    (for tests, experiments, etc.), use this method implementation as a test-case, i.e., override the
    'get_op_quantization_configs' method and use its output to call 'generate_hardware_model' with your configurations.

    Returns: A HardwareModel object.

    """
    base_config, mixed_precision_cfg_list = get_op_quantization_configs()
    return generate_hardware_model(base_config, mixed_precision_cfg_list, name='default_hwm')


def get_op_quantization_configs() -> Tuple[OpQuantizationConfig, List[OpQuantizationConfig]]:
    """
    Creates a default configuration object for 8-bit quantization, to be used to set a default hardware model.
    In addition, creates a default configuration objects list (with 8, 4 and 2 bit quantization) to be used as
    default configuration for mixed-precision quantization.

    Returns: An OpQuantizationConfig config object and a list of OpQuantizationConfig objects.

    """
    # Create a quantization config.
    # A quantization configuration defines how an operator
    # should be quantized on the modeled hardware:
    eight_bits = hwm.OpQuantizationConfig(
        activation_quantization_method=hwm.QuantizationMethod.POWER_OF_TWO,
        weights_quantization_method=hwm.QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        weights_n_bits=8,
        weights_per_channel_threshold=True,
        enable_weights_quantization=True,
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        weights_multiplier_nbits=None)

    # To quantize a model using mixed-precision, create
    # a list with more than one OpQuantizationConfig.
    # In this example, we quantize some operations' weights
    # using 2, 4 or 8 bits, and when using 2 or 4 bits, it's possible
    # to quantize the operations' activations using LUT.
    four_bits = eight_bits.clone_and_edit(weights_n_bits=4)
    two_bits = eight_bits.clone_and_edit(weights_n_bits=2)
    mixed_precision_cfg_list = [eight_bits, four_bits, two_bits]

    return eight_bits, mixed_precision_cfg_list


def generate_hardware_model(base_config: OpQuantizationConfig,
                            mixed_precision_cfg_list: List[OpQuantizationConfig],
                            name: str) -> HardwareModel:
    """
    Generates HardwareModel with default defined Operators Sets, based on the given base configuration and
    mixed-precision configurations options list.

    Args:
        base_config: An OpQuantizationConfig to set as the hardware model base configuration.
        mixed_precision_cfg_list: A list of OpQuantizationConfig to be used as the hardware model mixed-precision
            quantization configuration options.
        name: The name of the Hardware model.

    Returns: A HardwareModel object.

    """
    # Create a QuantizationConfigOptions, which defines a set
    # of possible configurations to consider when quantizing a set of operations (in mixed-precision, for example).
    # If the QuantizationConfigOptions contains only one configuration,
    # this configuration will be used for the operation quantization:
    default_configuration_options = hwm.QuantizationConfigOptions([base_config])

    # Create a HardwareModel and set its default quantization config.
    # This default configuration will be used for all operations
    # unless specified otherwise (see OperatorsSet, for example):
    generated_hwm = hwm.HardwareModel(default_configuration_options, name=name)

    # To start defining the model's components (such as operator sets, and fusing patterns),
    # use 'with' the hardware model instance, and create them as below:
    with generated_hwm:
        # Create an OperatorsSet to represent a set of operations.
        # Each OperatorsSet has a unique label.
        # If a quantization configuration options is passed, these options will
        # be used for operations that will be attached to this set's label.
        # Otherwise, it will be a configure-less set (used in fusing):

        # May suit for operations like: Dropout, Reshape, etc.
        hwm.OperatorsSet("NoQuantization",
                         hwm.get_default_quantization_config_options().clone_and_edit(
                             enable_weights_quantization=False,
                             enable_activation_quantization=False))

        # Create Mixed-Precision quantization configuration options from the given list of OpQuantizationConfig objects
        mixed_precision_configuration_options = hwm.QuantizationConfigOptions(mixed_precision_cfg_list,
                                                                              base_config=base_config)

        # Define operator sets that use mixed_precision_configuration_options:
        hwm.OperatorsSet("ConvTranspose", mixed_precision_configuration_options)
        conv = hwm.OperatorsSet("Conv", mixed_precision_configuration_options)
        fc = hwm.OperatorsSet("FullyConnected", mixed_precision_configuration_options)

        # Define operations sets without quantization configuration
        # options (useful for creating fusing patterns, for example):
        any_relu = hwm.OperatorsSet("AnyReLU")
        add = hwm.OperatorsSet("Add")
        prelu = hwm.OperatorsSet("PReLU")
        swish = hwm.OperatorsSet("Swish")
        sigmoid = hwm.OperatorsSet("Sigmoid")
        tanh = hwm.OperatorsSet("Tanh")

        # Combine multiple operators into a single operator to avoid quantization between
        # them. To do this we define fusing patterns using the OperatorsSets that were created.
        # To group multiple sets with regard to fusing, an OperatorSetConcat can be created
        activations_after_conv_to_fuse = hwm.OperatorSetConcat(any_relu, swish, prelu, sigmoid, tanh)
        activations_after_fc_to_fuse = hwm.OperatorSetConcat(any_relu, swish, sigmoid)

        hwm.Fusing([conv, activations_after_conv_to_fuse])
        hwm.Fusing([fc, activations_after_fc_to_fuse])
        hwm.Fusing([conv, add, any_relu])
        hwm.Fusing([conv, any_relu, add])

    return generated_hwm
