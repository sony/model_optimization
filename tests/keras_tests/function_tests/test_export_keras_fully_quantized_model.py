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
import random
import unittest

import keras.models
import numpy as np
from keras import Input
from keras.layers import Conv2D, BatchNormalization, ReLU, Dropout, Dense, Activation

import model_compression_toolkit as mct
from model_compression_toolkit.exporter.model_exporter import keras_export_model, KerasExportMode
from model_compression_toolkit.exporter.model_wrapper import is_keras_layer_exportable

SAVED_MODEL_PATH = '/tmp/exported_tf_fakelyquant.h5'


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
    outputs = Dense(10)(x)
    return keras.Model(inputs=inputs, outputs=outputs)


class TestKerasFakeQuantExporter(unittest.TestCase):

    def tearDown(self):
        os.remove(SAVED_MODEL_PATH)

    def setUp(self) -> None:
        input_shape = (32,32,3)
        self.model = _get_model(input_shape)
        self.representative_data_gen = lambda: [np.random.randn(*((1,)+input_shape))]
        self.exportable_model = self.run_mct(self.model, new_experimental_exporter=True)
        self.exported_model, self.custom_objects = keras_export_model(model=self.exportable_model,
                                                                      is_layer_exportable_fn=is_keras_layer_exportable,
                                                                      mode=KerasExportMode.FAKELY_QUANT,
                                                                      save_model_path=SAVED_MODEL_PATH)

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
        diff = self.exported_model(images) - self.exportable_model(images)
        print(f'Max abs error: {np.max(np.abs(diff))}')
        self.assertTrue(np.sum(np.abs(diff)) == 0)

    def test_load_model(self):
        """
        Test that the exported model (after loading it from file system) and fully quantized model predicting the
        same results.
        """
        loaded_model = keras.models.load_model(SAVED_MODEL_PATH, self.custom_objects)
        images = self.representative_data_gen()
        diff = loaded_model(images) - self.exportable_model(images)
        print(f'Max abs error: {np.max(np.abs(diff))}')
        self.assertTrue(np.sum(np.abs(diff)) == 0)
