# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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
import copy
import unittest
import numpy as np
from tensorflow.keras.applications.mobilenet import MobileNet
from model_compression_toolkit import get_keras_gptq_config, keras_post_training_quantization, DEFAULTCONFIG
import tensorflow as tf

def random_datagen():
    return [np.random.random((1, 224, 224, 3))]


class TestGetGPTQConfig(unittest.TestCase):

    def test_get_keras_gptq_config(self):
        qc = copy.deepcopy(DEFAULTCONFIG)
        qc.weights_bias_correction = False # disable bias correction when working with GPTQ

        gptq_configurations = [get_keras_gptq_config(n_iter=1),
                               get_keras_gptq_config(n_iter=1, train_bias=False),
                               get_keras_gptq_config(n_iter=1, optimizer=tf.keras.optimizers.RMSprop())]

        for gptq_config in gptq_configurations:
            keras_post_training_quantization(in_model=MobileNet(),
                                             representative_data_gen=random_datagen,
                                             n_iter=1,
                                             quant_config=qc,
                                             gptq_config=gptq_config)



if __name__ == '__main__':
    unittest.main()
