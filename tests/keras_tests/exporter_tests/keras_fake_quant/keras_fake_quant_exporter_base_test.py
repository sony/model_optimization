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

import keras.models
import numpy as np

import model_compression_toolkit as mct
from model_compression_toolkit import get_target_platform_capabilities, QuantizationConfig
from model_compression_toolkit.constants import TENSORFLOW
from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL


def get_minmax_from_qparams(qparams):
    if qparams['signed']:
        _min = -qparams['threshold'][0]
        _scale = qparams['threshold'][0] / (2 ** (qparams['num_bits'] - 1))
        _max = qparams['threshold'][0] - _scale
    else:
        _min = 0
        _scale = qparams['threshold'][0] / (2 ** qparams['num_bits'])
        _max = qparams['threshold'][0] - _scale

    return _min, _max


class KerasFakeQuantExporterBaseTest(ABC):

    def run_test(self):
        # Get float model and save it
        self.model = self.get_model()

        # Get fully quantized model
        self.exportable_model, _ = mct.ptq.keras_post_training_quantization_experimental(
            in_model=self.model,
            core_config=mct.core.CoreConfig(quantization_config=self.get_quantization_config()),
            representative_data_gen=self.__get_repr_dataset,
            target_platform_capabilities=self.get_tpc(),
            new_experimental_exporter=True)

        # Export model in h5 format
        _, self.fq_model_file_path = tempfile.mkstemp('.h5')
        mct.exporter.keras_export_model(model=self.exportable_model,
                                        save_model_path=self.fq_model_file_path,
                                        target_platform_capabilities=self.get_tpc())

        # Load model
        self.loaded_model = keras.models.load_model(self.fq_model_file_path)

        inputs = next(self.__get_repr_dataset())
        loaded_model_outputs = self.loaded_model(inputs)
        exportable_model_outputs = self.exportable_model(inputs)
        if not isinstance(loaded_model_outputs, list):
            loaded_model_outputs = [loaded_model_outputs]
        if not isinstance(exportable_model_outputs, list):
            exportable_model_outputs = [exportable_model_outputs]
        for loaded_out, exportable_out in zip(loaded_model_outputs, exportable_model_outputs):
            diff = np.sum(np.abs(loaded_out-exportable_out))
            assert diff == 0, f'Expected exportable model and exported model to have identical outputs but sum abs diff is {diff}'

        # Run tests
        self.run_checks()

        # Remove all files that was saved during the test run
        os.remove(self.fq_model_file_path)

    def get_input_shape(self):
        return [(16, 16, 3)]

    def get_tpc(self):
        return get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)

    def get_quantization_config(self):
        return QuantizationConfig()

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
