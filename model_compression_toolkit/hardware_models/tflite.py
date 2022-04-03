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
from model_compression_toolkit.common.hardware_representation.op_quantization_config import QuantizationMethod


def get_tflite_hw_model():

    # Create a quantization config. A quantization configuration defines how an operator
    # should be quantized on the modeled hardware. In TFLite
    # activations quantization is asymmetric, and weights quantization is symmetric:
    # https://www.tensorflow.org/lite/performance/quantization_spec#symmetric_vs_asymmetric
    eight_bits = hw_model.OpQuantizationConfig(
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

    # Create a QuantizationConfigOptions, which defines a set
    # of possible configurations to consider when quantizing a set of operations (in mixed-precision, for example).
    # If the QuantizationConfigOptions contains only one configuration,
    # this configuration will be used for the operation quantization:
    default_configuration_options = hw_model.QuantizationConfigOptions([eight_bits])

    # Create a HardwareModel and set its default quantization config.
    # This default configuration will be used for all operations
    # unless specified otherwise (see OperatorsSet, for example):
    tflite_model = hw_model.HardwareModel(default_configuration_options, name='tflite')

    # To start defining the model's components (such as operator sets, and fusing patterns),
    # use 'with' the hardware model instance, and create them as below:
    with tflite_model:
        # In TFLite, the quantized operator specifications constraint operators quantization
        # differently. For more details:
        # https://www.tensorflow.org/lite/performance/quantization_spec#int8_quantized_operator_specifications
        hw_model.OperatorsSet("PreserveQuantizationParams",
                              hw_model.get_default_quantization_config_options().clone_and_edit(
                                  quantization_preserving=True))

        fc = hw_model.OperatorsSet("FullyConnected",
                                   hw_model.get_default_quantization_config_options().clone_and_edit(
                                       weights_per_channel_threshold=False))

        hw_model.OperatorsSet("L2Normalization",
                              hw_model.get_default_quantization_config_options().clone_and_edit(
                                  fixed_zero_point=0, fixed_scale=1 / 128))
        hw_model.OperatorsSet("LogSoftmax",
                              hw_model.get_default_quantization_config_options().clone_and_edit(
                                  fixed_zero_point=127, fixed_scale=16 / 256))
        hw_model.OperatorsSet("Tanh",
                              hw_model.get_default_quantization_config_options().clone_and_edit(
                                  fixed_zero_point=0, fixed_scale=1 / 128))
        hw_model.OperatorsSet("Softmax",
                              hw_model.get_default_quantization_config_options().clone_and_edit(
                                  fixed_zero_point=-128, fixed_scale=1 / 256))
        hw_model.OperatorsSet("Logistic",
                              hw_model.get_default_quantization_config_options().clone_and_edit(
                                  fixed_zero_point=-128, fixed_scale=1 / 256))

        conv2d = hw_model.OperatorsSet("Conv2d")
        depthwise_conv2d = hw_model.OperatorsSet("DepthwiseConv2D")
        kernel = hw_model.OperatorSetConcat(conv2d, fc, depthwise_conv2d)

        relu = hw_model.OperatorsSet("Relu")
        elu = hw_model.OperatorsSet("Elu")
        activations_to_fuse = hw_model.OperatorSetConcat(relu, elu)

        batch_norm = hw_model.OperatorsSet("BatchNorm")
        bias_add = hw_model.OperatorsSet("BiasAdd")
        add = hw_model.OperatorsSet("Add")
        squeeze = hw_model.OperatorsSet("Squeeze",
                                        qc_options=hw_model.get_default_quantization_config_options().clone_and_edit(
                                            quantization_preserving=True))

        # Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/optimizers/remapper
        hw_model.Fusing([kernel, bias_add])
        hw_model.Fusing([kernel, bias_add, activations_to_fuse])
        hw_model.Fusing([conv2d, batch_norm, activations_to_fuse])
        hw_model.Fusing([conv2d, squeeze, activations_to_fuse])
        hw_model.Fusing([batch_norm, activations_to_fuse])
        hw_model.Fusing([batch_norm, add, activations_to_fuse])

    return tflite_model

