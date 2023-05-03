# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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
import os
import tempfile
import unittest

import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

import model_compression_toolkit as mct
from model_compression_toolkit import get_target_platform_capabilities
from model_compression_toolkit.constants import TENSORFLOW
from model_compression_toolkit.core.keras.constants import DEFAULT_TP_MODEL

DEFAULT_KERAS_TPC = get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)


class TestTFLiteExport(unittest.TestCase):

    def test_bn_folding_mbv1(self):
        model = MobileNetV2()

        def rep_data():
            yield [np.random.randn(1, 224, 224, 3)]

        self.tpc = DEFAULT_KERAS_TPC
        quantized_model, _ = mct.ptq. \
            keras_post_training_quantization_experimental(model,
                                                          rep_data,
                                                          target_platform_capabilities=self.tpc,
                                                          new_experimental_exporter=True)

        _, tflite_file_path = tempfile.mkstemp('.tflite')
        mct.exporter.keras_export_model(model=quantized_model,
                                        save_model_path=tflite_file_path,
                                        target_platform_capabilities=self.tpc,
                                        serialization_format=mct.exporter.ExportSerializationFormat.TFLITE)
        os.remove(tflite_file_path)


if __name__ == '__main__':
    unittest.main()
