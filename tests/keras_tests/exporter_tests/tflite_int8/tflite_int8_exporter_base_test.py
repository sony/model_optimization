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
from model_compression_toolkit.exporter.model_exporter import tflite_export_model, \
    TFLiteExportMode
from model_compression_toolkit.exporter.model_wrapper import is_keras_layer_exportable


class TFLiteINT8ExporterBaseTest:

    def run_test(self):
        # Get float model and save it
        self.model = self.get_model()
        _, self.float_model_file_path = tempfile.mkstemp('.h5')
        self.model.save(self.float_model_file_path)
        print(f'Float model was saved to: {self.float_model_file_path}')

        # Get fully quantized model
        self.exportable_model, _ = mct.keras_post_training_quantization_experimental(
            in_model=self.model,
            core_config=mct.CoreConfig(),
            representative_data_gen=self.__get_repr_dataset,
            new_experimental_exporter=True)

        # Disable bias as tflite quantizes them unlike MCT
        for l in self.exportable_model.layers:
            if hasattr(l, 'layer'):
                if hasattr(l.layer, 'use_bias'):
                    print(f'Disabling use_bias in {l.layer.name}')
                    l.layer.use_bias = False

        # Export model in INT8 format
        _, self.int8_model_file_path = tempfile.mkstemp('.tflite')
        tflite_export_model(model=self.exportable_model,
                            is_layer_exportable_fn=is_keras_layer_exportable,
                            mode=TFLiteExportMode.INT8,
                            save_model_path=self.int8_model_file_path)

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
        return [(224, 224, 3)]

    def __get_repr_dataset(self):
        for _ in range(1):
            yield [np.random.randn(*((1,) + shape)) for shape in self.get_input_shape()]

    def __infer_via_interpreter(self, inputs):
        input_index = self.interpreter.get_input_details()[0]["index"]
        self.interpreter.set_tensor(input_index, inputs.astype("float32"))
        # Run inference.
        self.interpreter.invoke()
        output_details = self.interpreter.get_output_details()
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
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
        assert np.all(self.interpreter.get_output_details()[0]['shape'][1:] == self.exportable_model.output_shape[1:]), \
            f'Expected shapes of exportable model and int8 tflite model are expected to be equal but exportable ' \
            f'output shape is {self.exportable_model.output_shape} and int8 output shape is ' \
            f'{self.interpreter.get_output_details()[0]["shape"]}'

        # Test inference and similarity to fully quantized model
        images = next(self.__get_repr_dataset())[0]
        pred_diff = self.exportable_model(images) - self.__infer_via_interpreter(images)
        assert np.sum(pred_diff != 0) / np.prod(
            pred_diff.shape) < 0.01, f'Parentage of different elements: ' \
                                     f'{np.sum(pred_diff != 0) / np.prod(pred_diff.shape)}'

        # Compare int8 model size to original float model
        float_model_size = os.path.getsize(self.float_model_file_path)
        int8_model_size = os.path.getsize(self.int8_model_file_path)
        assert float_model_size >= int8_model_size, f'INT8 model should be smaller than float model but INT8 model ' \
                                                    f'size is {int8_model_size} and float model size is {float_model_size}'
        print(f'Compression ratio: {float_model_size / int8_model_size}')

    # Helper method to create a keras model with a single layer
    def get_one_layer_model(self, layer):
        inputs = Input(shape=self.get_input_shape()[0])
        x = layer(inputs)
        return keras.Model(inputs=inputs, outputs=x)
