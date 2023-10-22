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
from model_compression_toolkit.target_platform_capabilities.target_platform import OpQuantizationConfig, \
    TargetPlatformModel
from model_compression_toolkit.target_platform_capabilities.target_platform.quantization_format import \
    QuantizationFormat

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
                             name='qnnpack_tp_model')


def get_op_quantization_configs() -> Tuple[OpQuantizationConfig, List[OpQuantizationConfig]]:
    """
    Creates a default configuration object for 8-bit quantization, to be used to set a default TargetPlatformModel.
    In addition, creates a default configuration objects list (with 8, 4 and 2 bit quantization) to be used as
    default configuration for mixed-precision quantization.

    Returns: An OpQuantizationConfig config object and a list of OpQuantizationConfig objects.

    """
    # Create a quantization config. A quantization configuration defines how an operator
    # should be quantized on the modeled hardware.
    # For qnnpack backend, Pytorch uses a QConfig with torch.per_tensor_affine for
    # activations quantization and a torch.per_tensor_symmetric quantization scheme
    # for weights quantization (https://pytorch.org/docs/stable/quantization.html#natively-supported-backends):
    eight_bits = tp.OpQuantizationConfig(
        activation_quantization_method=tp.QuantizationMethod.UNIFORM,
        weights_quantization_method=tp.QuantizationMethod.SYMMETRIC,
        activation_n_bits=8,
        weights_n_bits=8,
        weights_per_channel_threshold=False,
        enable_weights_quantization=True,
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        weights_multiplier_nbits=None,
        simd_size=None
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
    # use 'with' the target platform model instance, and create them as below:
    with generated_tpc:
        # Combine operations/modules into a single module.
        # Pytorch supports the next fusing patterns:
        # [Conv, Relu], [Conv, BatchNorm], [Conv, BatchNorm, Relu], [Linear, Relu]
        # Source: # https://pytorch.org/docs/stable/quantization.html#model-preparation-for-quantization-eager-mode
        conv = tp.OperatorsSet("Conv")
        batchnorm = tp.OperatorsSet("BatchNorm")
        relu = tp.OperatorsSet("Relu")
        linear = tp.OperatorsSet("Linear")

        # ------------------- #
        # Fusions
        # ------------------- #
        tp.Fusing([conv, batchnorm, relu])
        tp.Fusing([conv, batchnorm])
        tp.Fusing([conv, relu])
        tp.Fusing([linear, relu])

        # Set quantization format to fakely quant
        generated_tpc.set_quantization_format(QuantizationFormat.FAKELY_QUANT)

    return generated_tpc
