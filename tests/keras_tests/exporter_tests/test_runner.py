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

from tests.keras_tests.exporter_tests.tflite_fake_quant.networks.conv2d_test import TestConv2DTFLiteFQExporter, \
    TestConv2DReusedTFLiteFQExporter
from tests.keras_tests.exporter_tests.tflite_fake_quant.networks.dense_test import TestDenseReusedTFLiteFQExporter
from tests.keras_tests.exporter_tests.tflite_int8.networks.conv2d_test import TestConv2DPOTTFLiteINT8Exporter, TestConv2DSymmetricTFLiteINT8Exporter
from tests.keras_tests.exporter_tests.tflite_int8.networks.dense_test import TestDenseTFLiteINT8Exporter
from tests.keras_tests.exporter_tests.tflite_int8.networks.depthwiseconv2d_test import TestDepthwiseConv2DTFLiteINT8Exporter
from tests.keras_tests.exporter_tests.tflite_int8.networks.mobilenetv2_test import TestMBV2TFLiteINT8Exporter, \
    TestMBV2UniformActivationTFLiteINT8Exporter
from tests.keras_tests.function_tests.test_exporting_qat_models import TestExportingQATModelTFLite, TestExportingQATModelBase


class ExporterTestsRunner(unittest.TestCase):

    #################
    # TFLite INT8
    #################
    def test_conv2d(self):
        TestConv2DPOTTFLiteINT8Exporter().run_test()
        TestConv2DSymmetricTFLiteINT8Exporter().run_test()

    def test_depthwiseconv2d(self):
        TestDepthwiseConv2DTFLiteINT8Exporter().run_test()

    def test_dense(self):
        TestDenseTFLiteINT8Exporter().run_test()

    def test_mbv2(self):
        TestMBV2TFLiteINT8Exporter().run_test()
        TestMBV2UniformActivationTFLiteINT8Exporter().run_test()

    #####################
    # TFLite fake quant
    #####################

    def test_tflite_fq_conv2d(self):
        TestConv2DTFLiteFQExporter().run_test()

    def test_tflite_fq_conv2d_reused(self):
        TestConv2DReusedTFLiteFQExporter().run_test()

    def test_tflite_fq_dense_reused(self):
        TestDenseReusedTFLiteFQExporter().run_test()

    #########################
    # Exporting QAT models
    #########################

    def test_export_qat(self):
        TestExportingQATModelBase().test_exported_qat_model()
        TestExportingQATModelTFLite().test_exported_qat_model()

