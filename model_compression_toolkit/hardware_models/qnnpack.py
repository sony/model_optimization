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

from model_compression_toolkit import hardware_representation as hw_model


def get_qnnpack_model():
    # Create a quantization config. A quantization configuration defines how an operator
    # should be quantized on the modeled hardware.
    # For qnnpack backend, Pytorch uses a QConfig with torch.per_tensor_affine for
    # activations quantization and a torch.per_tensor_symmetric quantization scheme
    # for weights quantization (https://pytorch.org/docs/stable/quantization.html#natively-supported-backends):
    eight_bits = hw_model.OpQuantizationConfig(
        activation_quantization_method=hw_model.QuantizationMethod.UNIFORM,
        weights_quantization_method=hw_model.QuantizationMethod.SYMMETRIC,
        activation_n_bits=8,
        weights_n_bits=8,
        weights_per_channel_threshold=False,
        enable_weights_quantization=True,
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        weights_multiplier_nbits=None
    )

    # Create a QuantizationConfigOptions, which defines a set
    # of possible configurations to consider when quantizing a set of operations (in mixed-precision, for example).
    # If the QuantizationConfigOptions contains only one configuration,
    # this configuration will be used for the operation quantization:
    default_configuration_options = hw_model.QuantizationConfigOptions([eight_bits])

    # Create a HardwareModel and set its default quantization config.
    # This default configuration will be used for all operations
    # unless specified otherwise (see OperatorsSet, for example):
    qnnpack_model = hw_model.HardwareModel(default_configuration_options, name='qnnpack')

    # To start defining the model's components (such as operator sets, and fusing patterns),
    # use 'with' the hardware model instance, and create them as below:
    with qnnpack_model:
        # Combine operations/modules into a single module.
        # Pytorch supports the next fusing patterns:
        # [Conv, Relu], [Conv, BatchNorm], [Conv, BatchNorm, Relu], [Linear, Relu]
        # Source: # https://pytorch.org/docs/stable/quantization.html#model-preparation-for-quantization-eager-mode
        conv = hw_model.OperatorsSet("Conv")
        batchnorm = hw_model.OperatorsSet("BatchNorm")
        relu = hw_model.OperatorsSet("Relu")
        linear = hw_model.OperatorsSet("Linear")

        hw_model.Fusing([conv, batchnorm, relu])
        hw_model.Fusing([conv, batchnorm])
        hw_model.Fusing([conv, relu])
        hw_model.Fusing([linear, relu])

    return qnnpack_model

