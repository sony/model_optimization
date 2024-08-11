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

from model_compression_toolkit.verify_packages import FOUND_ONNXRUNTIME, FOUND_ONNX
from tests.pytorch_tests.exporter_tests.custom_ops_tests.test_export_lut_symmetric_onnx_quantizers import \
    TestExportONNXWeightLUTSymmetric2BitsQuantizers, TestExportONNXWeightLUTPOT2BitsQuantizers
from tests.pytorch_tests.exporter_tests.custom_ops_tests.test_export_pot_onnx_quantizers import \
    TestExportONNXWeightPOT2BitsQuantizers
from tests.pytorch_tests.exporter_tests.custom_ops_tests.test_export_symmetric_onnx_quantizers import \
    TestExportONNXWeightSymmetric2BitsQuantizers
from tests.pytorch_tests.exporter_tests.custom_ops_tests.test_export_uniform_onnx_quantizers import \
    TestExportONNXWeightUniform2BitsQuantizers


class PytorchExporterTestsRunner(unittest.TestCase):

    #########################
    # Exporting custom ONNX ops
    #########################
    def test_pot2bits_custom_quantizer_onnx(self):
        TestExportONNXWeightPOT2BitsQuantizers().run_test()
        TestExportONNXWeightPOT2BitsQuantizers(onnx_opset_version=16).run_test()

    def test_sym2bits_custom_quantizer_onnx(self):
        TestExportONNXWeightSymmetric2BitsQuantizers().run_test()
        TestExportONNXWeightSymmetric2BitsQuantizers(onnx_opset_version=16).run_test()



    def test_uniform2bits_custom_quantizer_onnx(self):
        TestExportONNXWeightUniform2BitsQuantizers().run_test()
        TestExportONNXWeightUniform2BitsQuantizers(onnx_opset_version=16).run_test()


    def test_lut_pot2bits_custom_quantizer_onnx(self):
        TestExportONNXWeightLUTPOT2BitsQuantizers().run_test()
        TestExportONNXWeightLUTPOT2BitsQuantizers(onnx_opset_version=16).run_test()


    def test_lut_sym2bits_custom_quantizer_onnx(self):
        TestExportONNXWeightLUTSymmetric2BitsQuantizers().run_test()
        TestExportONNXWeightLUTSymmetric2BitsQuantizers(onnx_opset_version=16).run_test()


