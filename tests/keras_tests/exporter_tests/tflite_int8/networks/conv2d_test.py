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
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.core.keras.constants import KERNEL
from model_compression_toolkit.target_platform_capabilities.tpc_models.default_tpc.v5.tpc_keras import generate_keras_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.keras_tests.exporter_tests.tflite_int8.tflite_int8_exporter_base_test import TFLiteINT8ExporterBaseTest
import keras
import numpy as np
import tests.keras_tests.exporter_tests.constants as constants
layers = keras.layers


class TestConv2DSymmetricTFLiteINT8Exporter(TFLiteINT8ExporterBaseTest):
    def __init__(self):
        super(TestConv2DSymmetricTFLiteINT8Exporter, self).__init__()
        self.weights_diff_tolerance=1e-7

    def get_model(self):
        return self.get_one_layer_model(layers.Conv2D(6, 5))

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_quantization_method': QuantizationMethod.SYMMETRIC})
        return generate_keras_tpc(name='sym_conv2d_exporter', tp_model=tp)


    def run_checks(self):
        # Fetch quantized weights from int8 model tensors
        kernel_quantization_parameters, kernel_tensor_index = None, None
        for t in self.interpreter.get_tensor_details():
            if np.all(t[constants.SHAPE] == np.asarray([6, 5, 5, 3])):
                kernel_tensor_index = t[constants.INDEX]
                kernel_quantization_parameters = t[constants.QUANTIZATION_PARAMETERS]
                print(kernel_quantization_parameters)
        assert kernel_quantization_parameters is not None
        assert kernel_tensor_index is not None

        # Assert there are 6 scales and zero points (like the number of output channels)
        assert len(kernel_quantization_parameters[constants.SCALES]) == 6
        assert len(kernel_quantization_parameters[constants.ZERO_POINTS]) == 6
        assert np.all(kernel_quantization_parameters[constants.ZERO_POINTS] == np.zeros(6))

        # Reshape Conv kernel to be at the same dimensions as in TF.
        kernel = self.interpreter.tensor(kernel_tensor_index)().transpose(1, 2, 3, 0)
        fake_quantized_kernel_from_exportable_model = self.exportable_model.layers[2].weights_quantizers[KERNEL](self.exportable_model.layers[2].layer.kernel)
        fake_quantized_kernel_from_int8_model = kernel * kernel_quantization_parameters[constants.SCALES].reshape(1, 1, 1, 6)
        max_abs_error = np.max(np.abs(fake_quantized_kernel_from_exportable_model-fake_quantized_kernel_from_int8_model))
        assert max_abs_error<=self.weights_diff_tolerance, f'Max abs diff between fake quant (from int8 model and exportable model) kernels passed tolerance: max_abs_error {max_abs_error}, tolerance:{self.weights_diff_tolerance}'


class TestConv2DPOTTFLiteINT8Exporter(TestConv2DSymmetricTFLiteINT8Exporter):

    def __init__(self):
        super(TestConv2DPOTTFLiteINT8Exporter, self).__init__()
        self.weights_diff_tolerance = 0

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_quantization_method': QuantizationMethod.POWER_OF_TWO})
        return generate_keras_tpc(name='sym_conv2d_exporter', tp_model=tp)

    def run_checks(self):
        super(TestConv2DPOTTFLiteINT8Exporter, self).run_checks()
        for tensor in self.interpreter.get_tensor_details():
            assert constants.QUANTIZATION_PARAMETERS in tensor.keys()
            scales = tensor[constants.QUANTIZATION_PARAMETERS][constants.SCALES]
            assert np.all(np.log2(scales) == np.round(np.log2(scales))), f'Expected all scales to be POT but scales are {scales} in tensor {tensor[constants.NAME]}'


