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

from tests.keras_tests.exporter_tests.tflite_int8.tflite_int8_exporter_base_test import TFLiteINT8ExporterBaseTest
import keras
layers = keras.layers

class TestMBV2Exporter(TFLiteINT8ExporterBaseTest):

    def get_input_shape(self):
        return [(224,224,3)]
    
    def get_model(self):
        return MobileNetV2()

    def run_checks(self):
        for tensor in self.interpreter.get_tensor_details():
            assert 'quantization_parameters' in tensor.keys()
            scales = tensor['quantization_parameters']['scales']
            assert np.all(np.log2(scales) == np.round(np.log2(scales))), f'Expected all scales to be POT but scales are {scales} in tensor {tensor["name"]}'

