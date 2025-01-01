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
from abc import abstractmethod

import keras.models
import numpy as np
import tensorflow as tf
from keras import Input

import model_compression_toolkit as mct
import tests.keras_tests.exporter_tests.constants as constants
from tests.keras_tests.exporter_tests.tflite_int8.imx500_int8_tp_model import get_int8_tpc


class TFLiteINT8ExporterBaseTest:

    def run_test(self):
        # Get float model and save it
        self.model = self.get_model()
        _, self.float_model_file_path = tempfile.mkstemp('.h5')
        self.model.save(self.float_model_file_path)
        print(f'Float model was saved to: {self.float_model_file_path}')

        # Get fully quantized model
        self.exportable_model, _ = mct.ptq.keras_post_training_quantization(
            in_model=self.model,
            core_config=mct.core.CoreConfig(),
            representative_data_gen=self.__get_repr_dataset,
            target_platform_capabilities=self.get_tpc())

        # Disable bias as tflite quantizes them unlike MCT
        for l in self.exportable_model.layers:
            if hasattr(l, 'layer'):
                if hasattr(l.layer, 'use_bias'):
                    print(f'Disabling use_bias in {l.layer.name}')
                    l.layer.use_bias = False

        # Export model in INT8 format
        _, self.int8_model_file_path = tempfile.mkstemp('.tflite')
        mct.exporter.keras_export_model(model=self.exportable_model,
                                        save_model_path=self.int8_model_file_path,
                                        serialization_format=mct.exporter.KerasExportSerializationFormat.TFLITE,
                                        quantization_format=mct.exporter.QuantizationFormat.INT8)

        # Load model
        self.interpreter = tf.lite.Interpreter(model_path=self.int8_model_file_path)
        self.interpreter.allocate_tensors()

        # Run tests
        self.run_common_checks()
        self.run_checks()

        # Remove all files that was saved during the test run
        os.remove(self.int8_model_file_path)
        os.remove(self.float_model_file_path)

    def get_input_shape(self):
        return [(16, 16, 3)]

    def get_tpc(self):
        return get_int8_tpc()

    def __get_repr_dataset(self):
        for _ in range(1):
            yield [np.random.randn(*((1,) + shape)) for shape in self.get_input_shape()]

    def __infer_via_interpreter(self, inputs):
        input_index = self.interpreter.get_input_details()[0][constants.INDEX]
        self.interpreter.set_tensor(input_index, inputs.astype("float32"))
        # Run inference.
        self.interpreter.invoke()
        output_details = self.interpreter.get_output_details()
        output_data = self.interpreter.get_tensor(output_details[0][constants.INDEX])
        return output_data

    @abstractmethod
    def get_model(self):
        raise Exception(f'Exporter test must implement get_model method')

    @abstractmethod
    def run_checks(self):
        """
        Tests can implement it for specific checks on the exported model
        """
        pass

    def run_common_checks(self):
        # Assert output shapes are the same
        # Ignore batch dim as it is constantly 1 in tflite models
        assert np.all(
            self.interpreter.get_output_details()[0][constants.SHAPE][1:] == self.exportable_model.output_shape[1:]), \
            f'Expected shapes of exportable model and int8 tflite model are expected to be equal but exportable ' \
            f'output shape is {self.exportable_model.output_shape} and int8 output shape is ' \
            f'{self.interpreter.get_output_details()[0][constants.SHAPE]}'

        # Test inference and similarity to fully quantized model
        images = next(self.__get_repr_dataset())[0]
        exportable_predictions = self.exportable_model(images)
        tflite_predictions = self.__infer_via_interpreter(images)

        # In order similarity between predictions we search for the last activation quantization delta.
        # We do this in order to assert that the outputs are similar up to the last scale which can
        # be caused due to rounding errors between the float and int8 models: https://github.com/tensorflow/tensorflow/issues/38845
        # The search assumes the delta of the last quantization node is in the last quantization
        # tensor details and it has a single scale
        scales = []
        for t in reversed(self.interpreter.get_tensor_details()):
            if len(t[constants.QUANTIZATION_PARAMETERS][constants.SCALES]) > 0:
                scales = t[constants.QUANTIZATION_PARAMETERS][constants.SCALES]
                break
        assert len(scales) == 1, f'Expected to find a single scale in the tensor details but scales are: {scales}'

        # Tolerance is an LSB due to rounding differences
        are_predictions_close = np.isclose(exportable_predictions, tflite_predictions, atol=scales[0])
        assert np.all(
            are_predictions_close), f'Outputs expected to be similar up to an LSB, but LSB is {scales[0]} and max error is {np.max(np.abs(exportable_predictions - tflite_predictions))}'

        # Compare int8 model size to original float model
        float_model_size = os.path.getsize(self.float_model_file_path)
        int8_model_size = os.path.getsize(self.int8_model_file_path)
        assert float_model_size >= int8_model_size, f'INT8 model should be smaller than float model but INT8 model ' \
                                                    f'size is {int8_model_size} bytes and float model size is {float_model_size} bytes'
        print(f'Compression ratio: {float_model_size / int8_model_size}')

    # Helper method to create a keras model with a single layer
    def get_one_layer_model(self, layer):
        inputs = Input(shape=self.get_input_shape()[0])
        x = layer(inputs)
        return keras.Model(inputs=inputs, outputs=x)
