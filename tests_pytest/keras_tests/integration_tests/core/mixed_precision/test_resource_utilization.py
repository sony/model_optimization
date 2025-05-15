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

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, Dense, Input, Subtract, Flatten, Add
from tensorflow.python.keras.layers import Activation

from tests_pytest._fw_tests_common_base.base_ru_integration_test import BaseRUIntegrationTester
from tests_pytest.keras_tests.keras_test_util.keras_test_mixin import KerasFwMixin


class TestRUIntegrationKeras(BaseRUIntegrationTester, KerasFwMixin):
    def test_compute_ru(self):
        super().test_compute_ru()

    def test_mult_output_activation(self):
        super().test_mult_output_activation()

    def test_snc_fusing(self):
        super().test_snc_fusing()

    def _data_gen(self):
        return self.get_basic_data_gen([self.bhwc_input_shape])()

    def _build_sequential_model(self):
        inputs = Input(shape=self.bhwc_input_shape[1:])
        x = Conv2D(filters=8, kernel_size=5)(inputs)
        x = tf.add(x, np.ones((14, 14, 8)))  # => activation with const in the composed node
        x = DepthwiseConv2D(kernel_size=3, depth_multiplier=2)(x)  # => Virtual activation in the composed node
        x = Conv2DTranspose(filters=12, kernel_size=5)(x)
        x = Flatten()(x)
        outputs = Dense(10)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def _build_mult_output_activation_model(self):
        inputs = Input(shape=self.bhwc_input_shape[1:])
        x1 = Conv2D(filters=15, kernel_size=3, groups=3)(inputs)
        x2 = DepthwiseConv2D(kernel_size=3, depth_multiplier=5)(inputs)
        x = Subtract()([x1, x2])
        x = Flatten()(x)
        outputs = Dense(10)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def _build_snc_model(self):
        inputs = Input(shape=self.bhwc_input_shape[1:])
        y = Conv2D(3, kernel_size=3, padding='same')(inputs)
        y = Activation('swish')(y)

        x = Add()([inputs, y])
        x = Conv2D(1, kernel_size=3)(x)
        x = Activation('swish')(x)
        x = Conv2D(2, kernel_size=3, padding='same')(x)
        outputs = Activation('swish')(x)
        return keras.Model(inputs=inputs, outputs=outputs)
