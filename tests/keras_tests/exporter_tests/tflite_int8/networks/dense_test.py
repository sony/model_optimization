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
from model_compression_toolkit.core.keras.constants import KERNEL
from tests.keras_tests.exporter_tests.tflite_int8.tflite_int8_exporter_base_test import TFLiteINT8ExporterBaseTest
import keras
import numpy as np
import tests.keras_tests.exporter_tests.constants as constants
from tests.keras_tests.utils import get_layers_from_model_by_type

layers = keras.layers

class TestDenseTFLiteINT8Exporter(TFLiteINT8ExporterBaseTest):

    def get_input_shape(self):
        # More than 3 dims input to test the substitution to point-wise
        return [(4,5,6,7,8)]

    def get_model(self):
        return self.get_one_layer_model(layers.Dense(20))

    def run_checks(self):
        # assert expected output shape
        expected_output_shape = np.asarray([1, 4, 5, 6, 7, 20])
        assert np.all(self.interpreter.get_output_details()[0][constants.SHAPE]==expected_output_shape), f'Expected output shape to be {expected_output_shape} but is {self.interpreter.get_output_details()[0][constants.SHAPE]}'

        # Fetch quantized weights from int8 model tensors
        kernel_quantization_parameters, kernel_tensor_index = None, None
        for t in self.interpreter.get_tensor_details():
            if len(t[constants.SHAPE]) == 4 and np.all(t[constants.SHAPE] == np.asarray([20, 1, 1, 8])):
                kernel_tensor_index = t[constants.INDEX]
                kernel_quantization_parameters = t[constants.QUANTIZATION_PARAMETERS]
        assert kernel_quantization_parameters is not None
        assert kernel_tensor_index is not None

        # Assert there are 20 scales and zero points (like the units number)
        assert len(kernel_quantization_parameters[constants.SCALES]) == 20
        assert len(kernel_quantization_parameters[constants.ZERO_POINTS]) == 20
        assert np.all(kernel_quantization_parameters[constants.ZERO_POINTS]==np.zeros(20))

        dense_layer = get_layers_from_model_by_type(self.exportable_model, layers.Dense)[0]
        fake_quantized_kernel_from_exportable_model = dense_layer.weights_quantizers[KERNEL](dense_layer.layer.kernel)
        # First reshape Conv kernel to be at the same dimensions as in TF.
        # Then reshape it to the original Dense kernel shape.
        # Then use scales to compute the fake quant kernel and compare it to the Dense fake quantized kernel
        kernel = self.interpreter.tensor(kernel_tensor_index)()
        fake_quantized_kernel_from_int8_model = kernel.transpose(1, 2, 3, 0).reshape(8, 20) * kernel_quantization_parameters[constants.SCALES].reshape(1, 20)
        assert np.all(fake_quantized_kernel_from_int8_model==fake_quantized_kernel_from_exportable_model), f'Expected quantized kernel to be the same in exportable model and in int8 model'

        for tensor in self.interpreter.get_tensor_details():
            assert constants.QUANTIZATION_PARAMETERS in tensor.keys()
            scales = tensor[constants.QUANTIZATION_PARAMETERS][constants.SCALES]
            assert np.all(np.log2(scales) == np.round(np.log2(scales))), f'Expected all scales to be POT but scales are {scales} in tensor {tensor[constants.NAME]}'
