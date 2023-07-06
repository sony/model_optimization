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
import keras
from keras import Input
from keras.layers import Conv2D
import numpy as np
import tests.keras_tests.exporter_tests.constants as constants


from tests.keras_tests.exporter_tests.tflite_fake_quant.tflite_fake_quant_exporter_base_test import \
    TFLiteFakeQuantExporterBaseTest

from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from model_compression_toolkit.target_platform_capabilities.tpc_models.default_tpc.latest import generate_keras_tpc


class TestConv2DTFLiteFQExporter(TFLiteFakeQuantExporterBaseTest):

    def get_input_shape(self):
        return [(30, 30, 3)]

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 2})
        return generate_keras_tpc(name="test_conv2d_2bit_fq_weight", tp_model=tp)

    def get_model(self):
        inputs = Input(shape=self.get_input_shape()[0])
        x = Conv2D(6, 20)(inputs)
        return keras.Model(inputs=inputs, outputs=x)

    def run_checks(self):
        # Fetch quantized weights from fq model tensors
        kernel_quantization_parameters, kernel_tensor_index, kernel_dtype = None, None, None
        for t in self.interpreter.get_tensor_details():
            if len(t[constants.SHAPE]) == 4 and np.all(t[constants.SHAPE] == np.asarray([6, 20, 20, 3])):
                kernel_tensor_index = t[constants.INDEX]
                kernel_quantization_parameters = t[constants.QUANTIZATION_PARAMETERS]
                kernel_dtype = t[constants.DTYPE]
        assert kernel_quantization_parameters is not None
        assert kernel_tensor_index is not None
        assert kernel_dtype is not None

        # Tensor should not be quantized (just fake-quantized)
        assert kernel_dtype == np.float32, f'Expected type of tensor to be float32 but is {kernel_dtype}'
        assert len(kernel_quantization_parameters[constants.SCALES]) == 0
        assert len(kernel_quantization_parameters[constants.ZERO_POINTS]) == 0
        kernel = self.interpreter.tensor(kernel_tensor_index)()
        for i in range(6):
            assert len(np.unique(kernel[i].flatten()))<=2**2, f'Each channel should have up to 4 unique values but filter {i} has {len(np.unique(kernel[i].flatten()))} unique values'



class TestConv2DReusedTFLiteFQExporter(TFLiteFakeQuantExporterBaseTest):

    def get_input_shape(self):
        return [(30, 30, 3)]

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 2})
        return generate_keras_tpc(name="test_conv2d_2bit_reused_weight", tp_model=tp)

    def get_model(self):
        conv = Conv2D(3,3)
        inputs = Input(shape=self.get_input_shape()[0])
        x = conv(inputs)
        x = conv(x)
        return keras.Model(inputs=inputs, outputs=x)

    def run_checks(self):
        ops = self.interpreter._get_ops_details()
        op_inputs = []
        for op in ops:
            if op['op_name'] == 'CONV_2D':
                op_inputs.append(op['inputs'])
        assert len(op_inputs) == 2, f'Expected to find 2 ops of CONV_2D but found {len(op_inputs)}'
        # Inputs like kernel and bias are expected to be the same since the conv is reused.
        first_inputs = op_inputs[0][1:] # Ignore first input of activations
        second_inputs = op_inputs[1][1:] # Ignore first input of activations
        assert np.all(first_inputs==second_inputs), f'Since conv is reused, the input weight tensors of the op are expected to be identical in both occurrences of the op but indices are {first_inputs} and {second_inputs}'
