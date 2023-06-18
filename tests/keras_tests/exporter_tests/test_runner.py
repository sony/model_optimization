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

from tests.keras_tests.exporter_tests.keras_fake_quant.networks.conv2d_test import TestConv2DKerasFQExporter, TestConv2DReusedKerasFQExporter, TestCon2DWeightsLUTKerasFQExporter
from tests.keras_tests.exporter_tests.keras_fake_quant.networks.conv2dtranspose_test import \
    TestConv2DTransposeKerasFQExporter
from tests.keras_tests.exporter_tests.keras_fake_quant.networks.dense_test import TestDenseKerasFQExporter
from tests.keras_tests.exporter_tests.keras_fake_quant.networks.dwconv2d_test import TestDWConv2DKerasFQExporter
from tests.keras_tests.exporter_tests.keras_fake_quant.networks.multiple_inputs_test import \
    TestMultipleInputsMultipleOutputsKerasFQExporter
from tests.keras_tests.exporter_tests.keras_fake_quant.networks.no_quant_test import TestNoQuantKerasFQExporter
from tests.keras_tests.exporter_tests.keras_fake_quant.networks.tfoplambda_test import TestTFOpLambdaKerasFQExporter


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
    # Keras fake quant
    #####################
    def test_keras_fq_conv2d(self):
        TestConv2DKerasFQExporter().run_test()
        TestConv2DReusedKerasFQExporter().run_test()

    def test_keras_fq_lut(self):
        TestCon2DWeightsLUTKerasFQExporter().run_test()

    def test_keras_fq_dwconv2d(self):
        TestDWConv2DKerasFQExporter().run_test()

    def test_keras_fq_dense(self):
        TestDenseKerasFQExporter().run_test()

    def test_keras_fq_conv2dtranspose(self):
        TestConv2DTransposeKerasFQExporter().run_test()

    def test_keras_fq_tfoplambda(self):
        TestTFOpLambdaKerasFQExporter().run_test()

    def test_keras_fq_multiplpe_inputs_multiple_outputs(self):
        TestMultipleInputsMultipleOutputsKerasFQExporter().run_test()

    def test_keras_fq_no_quant(self):
        TestNoQuantKerasFQExporter().run_test()


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

