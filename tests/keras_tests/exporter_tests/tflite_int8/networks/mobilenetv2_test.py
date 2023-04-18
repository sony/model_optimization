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

from keras.applications import MobileNetV2
import numpy as np

from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.tpc_models.default_tpc.latest import generate_keras_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.keras_tests.exporter_tests.tflite_int8.tflite_int8_exporter_base_test import TFLiteINT8ExporterBaseTest
import keras
import tests.keras_tests.exporter_tests.constants as constants

layers = keras.layers

class TestMBV2TFLiteINT8Exporter(TFLiteINT8ExporterBaseTest):

    def get_input_shape(self):
        return [(224,224,3)]
    
    def get_model(self):
        return MobileNetV2()

    def run_checks(self):
        for tensor in self.interpreter.get_tensor_details():
            assert constants.QUANTIZATION_PARAMETERS in tensor.keys()
            scales = tensor[constants.QUANTIZATION_PARAMETERS][constants.SCALES]
            assert np.all(np.log2(scales) == np.round(np.log2(scales))), f'Expected all scales to be POT but scales are {scales} in tensor {tensor[constants.NAME]}'


class TestMBV2UniformActivationTFLiteINT8Exporter(TestMBV2TFLiteINT8Exporter):

    def get_tpc(self):
        tp = generate_test_tp_model({'activation_quantization_method': QuantizationMethod.UNIFORM})
        return generate_keras_tpc(name='uniform_conv2d_exporter', tp_model=tp)

    def run_checks(self):
        pass


