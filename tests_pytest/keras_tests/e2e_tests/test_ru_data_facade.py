# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
import tensorflow as tf
import keras

from model_compression_toolkit.core import keras_resource_utilization_data
from tests_pytest._fw_tests_common_base.base_ru_data_facade_test import BaseRUDataFacadeTest


class TestKerasRUDataFacade(BaseRUDataFacadeTest):
    fw_ru_data_facade = keras_resource_utilization_data

    def _build_model(self, input_shape, out_chan, kernel, const):
        x = keras.layers.Input(input_shape[1:])
        y = keras.layers.Conv2D(filters=out_chan, kernel_size=kernel)(x)
        y = keras.layers.ReLU()(y)
        y = tf.add(y, const)
        return keras.Model(inputs=x, outputs=y)
