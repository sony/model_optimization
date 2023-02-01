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

import unittest

from tests.keras_tests.exporter_tests.tflite_int8.networks.conv2d_test import TestConv2DPOTTFLiteINT8Exporter, \
    TestConv2DSymmetricTFLiteINT8Exporter
from tests.keras_tests.exporter_tests.tflite_int8.networks.dense_test import TestDenseTFLiteINT8Exporter
from tests.keras_tests.exporter_tests.tflite_int8.networks.depthwiseconv2d_test import TestDepthwiseConv2DTFLiteINT8Exporter
from tests.keras_tests.exporter_tests.tflite_int8.networks.mobilenetv2_test import TestMBV2TFLiteINT8Exporter



class ExporterTestsRunner(unittest.TestCase):

    def test_conv2d(self):
        TestConv2DPOTTFLiteINT8Exporter().run_test()
        TestConv2DSymmetricTFLiteINT8Exporter().run_test()

    def test_depthwiseconv2d(self):
        TestDepthwiseConv2DTFLiteINT8Exporter().run_test()

    def test_dense(self):
        TestDenseTFLiteINT8Exporter().run_test()

    def test_mbv2(self):
        TestMBV2TFLiteINT8Exporter().run_test()




