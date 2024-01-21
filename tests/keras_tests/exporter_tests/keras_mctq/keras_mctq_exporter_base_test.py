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

import tempfile

from tests.keras_tests.exporter_tests.keras_fake_quant.keras_fake_quant_exporter_base_test import \
    KerasFakeQuantExporterBaseTest
import model_compression_toolkit as mct
import keras
import numpy as np
import os
from model_compression_toolkit.exporter.model_exporter.keras.base_keras_exporter import DEFAULT_KERAS_EXPORT_EXTENTION

class TestKerasMCTQExport(KerasFakeQuantExporterBaseTest):

    def __get_repr_dataset(self):
        yield [np.random.randn(*((1,) + shape)) for shape in self.get_input_shape()]

    def run_test(self):
        self.model = self.get_model()

        # Get fully quantized model
        self.exportable_model, _ = mct.ptq.keras_post_training_quantization_experimental(
            in_model=self.model,
            core_config=mct.core.CoreConfig(quantization_config=self.get_quantization_config()),
            representative_data_gen=self.__get_repr_dataset,
            target_platform_capabilities=self.get_tpc(),
            new_experimental_exporter=True)

        # Export model in keras (or h5) format
        with tempfile.NamedTemporaryFile(suffix=DEFAULT_KERAS_EXPORT_EXTENTION) as tmp_file:
            mct.exporter.keras_export_model(model=self.exportable_model,
                                            save_model_path=tmp_file.name,
                                            serialization_format=mct.exporter.KerasExportSerializationFormat.KERAS,
                                            quantization_format=mct.exporter.QuantizationFormat.MCTQ)

            # Load model
            self.loaded_model = mct.keras_load_quantized_model(tmp_file.name)

            inputs = next(self.__get_repr_dataset())
            loaded_model_outputs = self.loaded_model(inputs)
            exportable_model_outputs = self.exportable_model(inputs)
            if not isinstance(loaded_model_outputs, list):
                loaded_model_outputs = [loaded_model_outputs]
            if not isinstance(exportable_model_outputs, list):
                exportable_model_outputs = [exportable_model_outputs]
            for loaded_out, exportable_out in zip(loaded_model_outputs, exportable_model_outputs):
                diff = np.sum(np.abs(loaded_out - exportable_out))
                assert diff == 0, f'Expected exportable model and exported model to have identical outputs but sum abs diff is {diff}'

            # Run tests
            self.run_checks()

    def run_checks(self):
        # Each test should check different things
        pass
