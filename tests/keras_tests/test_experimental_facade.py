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
from keras.models import save_model

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
if tf.__version__ < "2.6":
    pass
else:
    pass

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.constants import TENSORFLOW
from model_compression_toolkit.core.common.target_platform.targetplatform2framework import LayerFilterParams
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    DEFAULT_MIXEDPRECISION_CONFIG
from model_compression_toolkit.core.keras.constants import DEFAULT_TP_MODEL, QNNPACK_TP_MODEL, TFLITE_TP_MODEL
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation

tp = mct.target_platform


class TestGPTQExperimentalFacade(unittest.TestCase):

    def set_seed(self, seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

    def test_sanity(self):
        repr_dataset = lambda: [np.random.randn(1, 224, 224, 3)]
        for seed in range(10):
            self.set_seed(seed)
            old_export_model, _ = mct.keras_post_training_quantization(in_model=MobileNetV2(),
                                                                       representative_data_gen=repr_dataset,
                                                                       n_iter=1,
                                                                       gptq_config=mct.get_keras_gptq_config(n_iter=1)
                                                                       )

            self.set_seed(seed)
            core_config = mct.CoreConfig(n_iter=1)
            new_export_model, _ = mct.keras_gradient_post_training_quantization_experimental(in_model=MobileNetV2(),
                                                                                             core_config=core_config,
                                                                                             representative_data_gen=repr_dataset,
                                                                                             gptq_config=mct.get_keras_gptq_config(n_iter=1))


            images = repr_dataset()
            diff = new_export_model(images) - old_export_model(images)
            print(f'Seed: {seed}')
            print(f'Max abs error: {np.max(np.abs(diff))}')
            self.assertTrue(np.sum(np.abs(diff))==0)


class TestPTQExperimentalFacade(unittest.TestCase):

    def set_seed(self, seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

    def test_sanity(self):
        repr_dataset = lambda: [np.random.randn(1, 224, 224, 3)]
        for seed in range(10):
            self.set_seed(seed)
            old_export_model, _ = mct.keras_post_training_quantization(in_model=MobileNetV2(),
                                                                       representative_data_gen=repr_dataset,
                                                                       n_iter=1
                                                                       )

            self.set_seed(seed)
            core_config = mct.CoreConfig(n_iter=1)
            new_export_model, _ = mct.keras_post_training_quantization_experimental(in_model=MobileNetV2(),
                                                                                             core_config=core_config,
                                                                                             representative_data_gen=repr_dataset)


            images = repr_dataset()
            diff = new_export_model(images) - old_export_model(images)
            print(f'Seed: {seed}')
            print(f'Max abs error: {np.max(np.abs(diff))}')
            self.assertTrue(np.sum(np.abs(diff))==0)
