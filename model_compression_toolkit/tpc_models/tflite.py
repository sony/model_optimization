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

from model_compression_toolkit import target_platform as tpc
from model_compression_toolkit.common.target_platform.op_quantization_config import QuantizationMethod


def get_tflite_tp_model():
    # Create a quantization config. A quantization configuration defines how an operator
    # should be quantized on the modeled hardware. In TFLite
    # activations quantization is asymmetric, and weights quantization is symmetric:
    # https://www.tensorflow.org/lite/performance/quantization_spec#symmetric_vs_asymmetric
    eight_bits = tpc.OpQuantizationConfig(
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
    default_configuration_options = tpc.QuantizationConfigOptions([eight_bits])

    # Create a TargetPlatformModel and set its default quantization config.
    # This default configuration will be used for all operations
    # unless specified otherwise (see OperatorsSet, for example):
    tflite_model = tpc.TargetPlatformModel(default_configuration_options, name='tflite')

    # To start defining the model's components (such as operator sets, and fusing patterns),
    # use 'with' the target platform model instance, and create them as below:
    with tflite_model:
        # In TFLite, the quantized operator specifications constraint operators quantization
        # differently. For more details:
        # https://www.tensorflow.org/lite/performance/quantization_spec#int8_quantized_operator_specifications
        tpc.OperatorsSet("PreserveQuantizationParams",
                         tpc.get_default_quantization_config_options().clone_and_edit(
                                  quantization_preserving=True))

        fc = tpc.OperatorsSet("FullyConnected",
                              tpc.get_default_quantization_config_options().clone_and_edit(
                                       weights_per_channel_threshold=False))

        tpc.OperatorsSet("L2Normalization",
                         tpc.get_default_quantization_config_options().clone_and_edit(
                                  fixed_zero_point=0, fixed_scale=1 / 128))
        tpc.OperatorsSet("LogSoftmax",
                         tpc.get_default_quantization_config_options().clone_and_edit(
                                  fixed_zero_point=127, fixed_scale=16 / 256))
        tpc.OperatorsSet("Tanh",
                         tpc.get_default_quantization_config_options().clone_and_edit(
                                  fixed_zero_point=0, fixed_scale=1 / 128))
        tpc.OperatorsSet("Softmax",
                         tpc.get_default_quantization_config_options().clone_and_edit(
                                  fixed_zero_point=-128, fixed_scale=1 / 256))
        tpc.OperatorsSet("Logistic",
                         tpc.get_default_quantization_config_options().clone_and_edit(
                                  fixed_zero_point=-128, fixed_scale=1 / 256))

        conv2d = tpc.OperatorsSet("Conv2d")
        depthwise_conv2d = tpc.OperatorsSet("DepthwiseConv2D")
        kernel = tpc.OperatorSetConcat(conv2d, fc, depthwise_conv2d)

        relu = tpc.OperatorsSet("Relu")
        elu = tpc.OperatorsSet("Elu")
        activations_to_fuse = tpc.OperatorSetConcat(relu, elu)

        batch_norm = tpc.OperatorsSet("BatchNorm")
        bias_add = tpc.OperatorsSet("BiasAdd")
        add = tpc.OperatorsSet("Add")
        squeeze = tpc.OperatorsSet("Squeeze",
                                   qc_options=tpc.get_default_quantization_config_options().clone_and_edit(quantization_preserving=True))

        # Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/optimizers/remapper
        tpc.Fusing([kernel, bias_add])
        tpc.Fusing([kernel, bias_add, activations_to_fuse])
        tpc.Fusing([conv2d, batch_norm, activations_to_fuse])
        tpc.Fusing([conv2d, squeeze, activations_to_fuse])
        tpc.Fusing([batch_norm, activations_to_fuse])
        tpc.Fusing([batch_norm, add, activations_to_fuse])

    return tflite_model

