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

import os
import tempfile
from abc import abstractmethod, ABC

import numpy as np
import tensorflow as tf

import model_compression_toolkit as mct
from model_compression_toolkit import get_target_platform_capabilities
from model_compression_toolkit.core.common.constants import TENSORFLOW
from model_compression_toolkit.core.keras.constants import DEFAULT_TP_MODEL


class TFLiteFakeQuantExporterBaseTest(ABC):

    def run_test(self):
        # Get float model and save it
        self.model = self.get_model()
        _, self.float_model_file_path = tempfile.mkstemp('.h5')
        self.model.save(self.float_model_file_path)
        print(f'Float model was saved to: {self.float_model_file_path}')

        # Get fully quantized model
        self.exportable_model, _ = mct.ptq.keras_post_training_quantization_experimental(
            in_model=self.model,
            core_config=mct.CoreConfig(),
            representative_data_gen=self.__get_repr_dataset,
            target_platform_capabilities=self.get_tpc(),
            new_experimental_exporter=True)

        # Export model in fake-quantized format
        _, self.fq_model_file_path = tempfile.mkstemp('.tflite')
        mct.exporter.tflite_export_model(model=self.exportable_model,
                                         mode=mct.exporter.TFLiteExportMode.FAKELY_QUANT,
                                         save_model_path=self.fq_model_file_path)

        # Load model
        self.interpreter = tf.lite.Interpreter(model_path=self.fq_model_file_path)
        self.interpreter.allocate_tensors()

        # Run tests
        self.run_checks()

        # Remove all files that was saved during the test run
        os.remove(self.fq_model_file_path)
        os.remove(self.float_model_file_path)

    def get_input_shape(self):
        return [(16, 16, 3)]

    def get_tpc(self):
        return get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)

    def __get_repr_dataset(self):
        yield [np.random.randn(*((1,) + shape)) for shape in self.get_input_shape()]

    @abstractmethod
    def get_model(self):
        raise Exception(f'Exporter test must implement get_model method')

    @abstractmethod
    def run_checks(self):
        """
        Tests must implement it for specific checks on the exported model
        """
        raise Exception(f'Exporter test must implement run_checks method')
