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


import tensorflow as tf

from model_compression_toolkit.constants import THRESHOLD
from model_compression_toolkit.core.keras.constants import KERNEL
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers


class PerTensorWeightQuantizationTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test )

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_per_channel_threshold': False})
        return generate_keras_tpc(name="per_tensor_weight_quantization", tp_model=tp)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(6, 7)(inputs)
        return keras.Model(inputs=inputs, outputs=x)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        conv_layer = get_layers_from_model_by_type(quantized_model, layers.Conv2D)[0]
        self.unit_test.assertTrue(
            len(conv_layer.weights_quantizers[KERNEL].get_config()[THRESHOLD]) == 1,
            f'Expected in per-tensor quantization to have a single threshold but found '
            f'{len(conv_layer.weights_quantizers[KERNEL].get_config()[THRESHOLD])}')
