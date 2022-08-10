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
from typing import List
import tensorflow as tf

import model_compression_toolkit as mct
from tests.keras_tests.feature_networks_tests.feature_networks.gptq.gptq_test import GradientPTQLearnRateZeroTest, \
    GradientPTQWeightsUpdateTest

keras = tf.keras
layers = keras.layers
tp = mct.target_platform


def build_model(in_input_shape: List[int], group: int = 1, dilation_rate=(1, 1)) -> keras.Model:
    """
    This function generate a simple network to test GPTQ
    Args:
        in_input_shape: Input shape list
        group: convulation group parameter

    Returns:

    """
    inputs = layers.Input(shape=in_input_shape)
    x = layers.Conv2D(16, 4, bias_initializer='glorot_uniform', dilation_rate=dilation_rate)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.Conv2D(16, 8, bias_initializer='glorot_uniform', groups=group)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.ReLU()(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


class GradientPTQWeightsUpdateConvGroupTest(GradientPTQWeightsUpdateTest):
    def create_networks(self):
        in_shape = self.get_input_shapes()[0][1:]
        return build_model(in_shape, group=2)


class GradientPTQLearnRateZeroConvGroupTest(GradientPTQLearnRateZeroTest):
    def create_networks(self):
        in_shape = self.get_input_shapes()[0][1:]
        return build_model(in_shape, group=2)


class GradientPTQWeightsUpdateConvGroupDilationTest(GradientPTQWeightsUpdateTest):
    def create_networks(self):
        in_shape = self.get_input_shapes()[0][1:]
        return build_model(in_shape, group=1, dilation_rate=(2, 2))


class GradientPTQLearnRateZeroConvGroupDilationTest(GradientPTQLearnRateZeroTest):
    def create_networks(self):
        in_shape = self.get_input_shapes()[0][1:]
        return build_model(in_shape, group=1, dilation_rate=(2, 2))
