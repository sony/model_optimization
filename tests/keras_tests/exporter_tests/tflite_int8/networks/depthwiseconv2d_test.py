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
from model_compression_toolkit.core.keras.constants import DEPTHWISE_KERNEL
from tests.keras_tests.exporter_tests.tflite_int8.tflite_int8_exporter_base_test import TFLiteINT8ExporterBaseTest
import keras
import numpy as np
import tests.keras_tests.exporter_tests.constants as constants

layers = keras.layers

class TestDepthwiseConv2DTFLiteINT8Exporter(TFLiteINT8ExporterBaseTest):
    def get_model(self):
        return self.get_one_layer_model(layers.DepthwiseConv2D(9))

    def run_checks(self):
        # Fetch quantized weights from int8 model tensors
        kernel_quantization_parameters, kernel_tensor_index = None, None
        for t in self.interpreter.get_tensor_details():
            if np.all(t[constants.SHAPE] == np.asarray([9, 9, 1, 3])):
                kernel_tensor_index = t[constants.INDEX]
                kernel_quantization_parameters = t[constants.QUANTIZATION_PARAMETERS]

        assert kernel_quantization_parameters is not None
        assert kernel_tensor_index is not None

        # Assert there are 3 scales and zero points (like the number of output channels)
        assert len(kernel_quantization_parameters[constants.SCALES]) == 3

        assert len(kernel_quantization_parameters[constants.ZERO_POINTS]) == 3
        assert np.all(kernel_quantization_parameters[constants.ZERO_POINTS] == np.zeros(3))

        # Reshape DW kernel to be at the same dimensions as in TF.
        kernel = self.interpreter.tensor(kernel_tensor_index)().transpose(0, 1, 3, 2)
        fake_quantized_kernel_from_exportable_model = self.exportable_model.layers[2].weights_quantizers[DEPTHWISE_KERNEL](self.exportable_model.layers[2].layer.depthwise_kernel)
        fake_quantized_kernel_from_int8_model = kernel * kernel_quantization_parameters[constants.SCALES].reshape(1, 1, 3, 1)
        assert np.all(
            fake_quantized_kernel_from_exportable_model == fake_quantized_kernel_from_int8_model), f'Expected quantized kernel to be the same in exportable model and in int8 model'

        for tensor in self.interpreter.get_tensor_details():
            assert constants.QUANTIZATION_PARAMETERS in tensor.keys()
            scales = tensor[constants.QUANTIZATION_PARAMETERS][constants.SCALES]
            assert np.all(np.log2(scales) == np.round(np.log2(scales))), f'Expected all scales to be POT but scales are {scales} in tensor {tensor[constants.NAME]}'
