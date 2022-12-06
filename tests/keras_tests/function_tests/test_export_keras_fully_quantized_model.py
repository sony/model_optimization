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
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

import model_compression_toolkit as mct
from model_compression_toolkit.exporter.target_platform_export.keras import export_keras_fully_quantized_model, \
    KerasExportMode
from tests.keras_tests.tpc_keras import get_activation_quantization_disabled_keras_tpc

tp = mct.target_platform

SAVED_MODEL_PATH = '/tmp/exported_tf_fakelyquant.h5'


class TestExporter(unittest.TestCase):

    def tearDown(self):
        os.remove(SAVED_MODEL_PATH)

    def setUp(self) -> None:
        self.mbv2 = MobileNetV2()
        self.representative_data_gen = lambda: [np.random.randn(1, 224, 224, 3)]
        self.fully_quantized_mbv2 = self.run_mct(self.mbv2)
        self.exported_mbv2 = export_keras_fully_quantized_model(model=self.fully_quantized_mbv2,
                                                                mode=KerasExportMode.FAKELY_QUANT,
                                                                save_model_path=SAVED_MODEL_PATH)

    def run_mct(self, model):
        core_config = mct.CoreConfig()
        tpc = get_activation_quantization_disabled_keras_tpc('test_keras_exporter')

        new_export_model, _ = mct.keras_post_training_quantization_experimental(
            in_model=model,
            core_config=core_config,
            target_platform_capabilities=tpc,
            representative_data_gen=self.representative_data_gen,
            new_experimental_exporter=True)
        return new_export_model

    def set_seed(self, seed):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def test_sanity(self):
        """
        Test that the exported model and fully quantized model predicting the same results.
        """
        images = self.representative_data_gen()
        diff = self.exported_mbv2(images) - self.fully_quantized_mbv2(images)
        print(f'Max abs error: {np.max(np.abs(diff))}')
        self.assertTrue(np.sum(np.abs(diff)) == 0)

    def test_load_model(self):
        """
        Test that the exported model (after loading it from file system) and fully quantized model predicting the
        same results.
        """
        loaded_model = keras.models.load_model(SAVED_MODEL_PATH)
        images = self.representative_data_gen()
        diff = loaded_model(images) - self.fully_quantized_mbv2(images)
        print(f'Max abs error: {np.max(np.abs(diff))}')
        self.assertTrue(np.sum(np.abs(diff)) == 0)
