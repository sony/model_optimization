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

from model_compression_toolkit.constants import FOUND_ONNXRUNTIME, FOUND_ONNX
from tests.pytorch_tests.exporter_tests.test_exporting_qat_models import TestExportingQATModelTorchscript


class PytorchExporterTestsRunner(unittest.TestCase):

    #########################
    # Exporting QAT models
    #########################

    def test_export_qat(self):
        TestExportingQATModelTorchscript().test_exported_qat_model()
        if FOUND_ONNX and FOUND_ONNXRUNTIME:
            from tests.pytorch_tests.exporter_tests.test_exporting_qat_models import TestExportingQATModelONNX
            TestExportingQATModelONNX().test_exported_qat_model()

