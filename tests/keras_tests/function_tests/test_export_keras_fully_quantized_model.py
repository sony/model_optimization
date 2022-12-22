# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
import unittest

import keras.models
import numpy as np
import tensorflow as tf
from keras import Input
from keras.layers import Conv2D, BatchNormalization, ReLU, Dropout, Dense, Activation

import model_compression_toolkit as mct
from model_compression_toolkit.exporter.model_exporter import keras_export_model, KerasExportMode, \
    tflite_export_model, \
    TFLiteExportMode
from model_compression_toolkit.exporter.model_wrapper import is_keras_layer_exportable

_, SAVED_MODEL_PATH_TF = tempfile.mkstemp('.h5')
_, SAVED_MODEL_PATH_TFLITE = tempfile.mkstemp('.tflite')


def _get_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(3, 3)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(3, 3)(x)
    x = Activation('swish')(x)
    x = Conv2D(3, 3, padding='same')(x)
    x = ReLU(max_value=6)(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(10)(x)
    return keras.Model(inputs=inputs, outputs=x)


class TestKerasFakeQuantExporter(unittest.TestCase):

    def tearDown(self):
        os.remove(SAVED_MODEL_PATH_TF)
        os.remove(SAVED_MODEL_PATH_TFLITE)

    def setUp(self) -> None:
        input_shape = (32, 32, 3)
        self.model = _get_model(input_shape)
        self.representative_data_gen = lambda: [np.random.randn(*((1,) + input_shape))]
        self.exportable_model = self.run_mct(self.model, new_experimental_exporter=True)
        self.exportable_model.trainable = False
        self.tf_custom_objects = keras_export_model(model=self.exportable_model,
                                                    is_layer_exportable_fn=is_keras_layer_exportable,
                                                    mode=KerasExportMode.FAKELY_QUANT,
                                                    save_model_path=SAVED_MODEL_PATH_TF)

        self.tf_exported_model = keras.models.load_model(SAVED_MODEL_PATH_TF, self.tf_custom_objects)
        self.tf_exported_model.trainable = False
        tflite_export_model(model=self.exportable_model,
                            is_layer_exportable_fn=is_keras_layer_exportable,
                            mode=TFLiteExportMode.FAKELY_QUANT,
                            save_model_path=SAVED_MODEL_PATH_TFLITE)

    def run_mct(self, model, new_experimental_exporter):
        core_config = mct.CoreConfig()
        new_export_model, _ = mct.keras_post_training_quantization_experimental(
            in_model=model,
            core_config=core_config,
            representative_data_gen=self.representative_data_gen,
            new_experimental_exporter=new_experimental_exporter)
        return new_export_model

    def test_sanity(self):
        """
        Test that the exported model and fully quantized model predicting the same results.
        """
        images = self.representative_data_gen()
        diff = self.tf_exported_model(images) - self.exportable_model(images)
        print(f'Max abs error: {np.max(np.abs(diff))}')
        self.assertTrue(np.sum(np.abs(diff)) == 0)

    def test_load_model(self):
        """
        Test that the exported model (after loading it from file system) and fully quantized model predicting the
        same results.
        """
        images = self.representative_data_gen()
        diff = self.tf_exported_model(images) - self.exportable_model(images)
        print(f'Max abs error: {np.max(np.abs(diff))}')
        self.assertTrue(np.sum(np.abs(diff)) == 0)

    def test_tflite_fq_export(self):
        """
        Test that the tflite exported model can infer and that tf exported model has the same weights
        as the tflite exported model.
        """
        # Test inference of exported model
        test_image = self.representative_data_gen()[0].astype("float32")
        interpreter = tf.lite.Interpreter(model_path=SAVED_MODEL_PATH_TFLITE)
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]["index"]
        interpreter.set_tensor(input_index, test_image)
        # Run inference.
        interpreter.invoke()

        # Test equal weights of the first conv2d layer between exported TFLite and exported TF
        diff = np.transpose(interpreter.tensor(4)(), (1, 2, 3, 0)) - self.tf_exported_model.layers[2].kernel
        print(f'Max abs error: {np.max(np.abs(diff))}')
        self.assertTrue(np.sum(np.abs(diff)) == 0)

        # Make sure the bias is equal
        diff = interpreter.tensor(1)() - self.tf_exported_model.layers[2].bias
        print(f'Max abs error: {np.max(np.abs(diff))}')
        self.assertTrue(np.sum(np.abs(diff)) == 0)
