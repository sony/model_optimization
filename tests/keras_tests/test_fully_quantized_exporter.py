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
import random
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

import model_compression_toolkit as mct

tp = mct.target_platform


class TestFullyQuantizedExporter(unittest.TestCase):

    def set_seed(self, seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

    def test_sanity(self):
        """
        Test that new fully quantized exporter model outputs the same as
        old exported model.
        """
        repr_dataset = lambda: [np.random.randn(1, 224, 224, 3)]
        seed = np.random.randint(0, 100, size=1)[0]

        self.set_seed(seed)
        core_config = mct.CoreConfig(n_iter=1)
        old_export_model, _ = mct.keras_post_training_quantization_experimental(in_model=MobileNetV2(),
                                                                                representative_data_gen=repr_dataset,
                                                                                core_config=core_config)

        self.set_seed(seed)
        core_config = mct.CoreConfig(n_iter=1)
        new_export_model, _ = mct.keras_post_training_quantization_experimental(in_model=MobileNetV2(),
                                                                                core_config=core_config,
                                                                                representative_data_gen=repr_dataset,
                                                                                new_experimental_exporter=True)

        images = repr_dataset()
        diff = new_export_model(images) - old_export_model(images)
        print(f'Max abs error: {np.max(np.abs(diff))}')
        self.assertTrue(np.sum(np.abs(diff)) == 0)
