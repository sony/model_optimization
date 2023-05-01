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
import tensorflow as tf


class TestTFLiteExport(unittest.TestCase):

    def test_bn_folding_mbv1(self):
        model = MobileNetV2()

        def rep_data():
            yield [np.random.randn(1, 224, 224, 3)]

        quantized_model, _ = mct.ptq.keras_post_training_quantization_experimental(model, rep_data, new_experimental_exporter=True)

        _, tflite_file_path = tempfile.mkstemp('.tflite')
        mct.exporter.tflite_export_model(quantized_model, tflite_file_path, mct.exporter.TFLiteExportMode.FAKELY_QUANT)
        os.remove(tflite_file_path)


if __name__ == '__main__':
    unittest.main()
