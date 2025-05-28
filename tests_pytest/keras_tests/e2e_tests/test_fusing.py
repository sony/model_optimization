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


from model_compression_toolkit import get_target_platform_capabilities
from model_compression_toolkit.core.keras.resource_utilization_data_facade import keras_resource_utilization_data
from model_compression_toolkit.constants import TENSORFLOW
from model_compression_toolkit.ptq import keras_post_training_quantization
from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL

from tensorflow.keras import layers, models

from tests_pytest._fw_tests_common_base.base_fusing_test import BaseFusingTest
from mct_quantizers import KerasActivationQuantizationHolder
from tests_pytest.keras_tests.keras_test_util.keras_test_mixin import KerasFwMixin

class TestKerasFusing(BaseFusingTest, KerasFwMixin):

    bhwc_input_shape = (1, 18, 18, 3)

    fw_ptq_facade = keras_post_training_quantization
    tpc = get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)
    fw_ru_data_facade = keras_resource_utilization_data

    def _build_test_model_basic_fusing(self, input_shape):
        inputs = layers.Input(shape=input_shape[1:])
        x = layers.Conv2D(3, 3)(inputs)
        x = layers.ReLU()(x)
        x = layers.Conv2D(2, 3)(x)
        x = layers.Activation('sigmoid')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(10)(x)
        x = layers.Activation('swish')(x)
        return models.Model(inputs=inputs, outputs=x)

    def _build_test_model_reuse(self, input_shape):
        inputs = layers.Input(shape=input_shape[1:])
        conv = layers.Conv2D(3, 3)
        x = conv(inputs)
        x = layers.ReLU()(x)
        x = conv(x)
        x = layers.ReLU()(x)
        return models.Model(inputs=inputs, outputs=x)

    def _build_test_model_ru_data_facade(self, input_shape):
        inputs = layers.Input(shape=input_shape[1:])
        x = layers.Conv2D(3, 1)(inputs)
        x = layers.ReLU()(x)
        x = layers.Add()([x, inputs])
        return models.Model(inputs=inputs, outputs=x)

    def _build_test_model_snc(self, input_shape):
        inputs = layers.Input(shape=input_shape[1:])
        x = layers.Conv2D(3, 3, padding='same')(inputs)
        x = layers.Activation('swish')(x)
        x = layers.Add()([x, inputs])
        x = layers.Conv2D(1, 3)(x)
        x = layers.Activation('swish')(x)
        x = layers.Conv2D(2, 3, padding='same')(x)
        x = layers.Activation('swish')(x)
        return models.Model(inputs=inputs, outputs=x)

    def _get_expected_act_quant_holders(self):
        return ['activation_holder_quantizer',
                're_lu_activation_holder_quantizer',
                'activation_activation_holder_quantizer',
                'activation_1_activation_holder_quantizer']

    def _get_expected_act_quant_holders_in_reuse_model(self):
        return ['activation_holder_quantizer',
                're_lu_activation_holder_quantizer',
                're_lu_1_activation_holder_quantizer']

    def _get_actual_act_quant_holders(self, qmodel):
        return [layer.name for layer in qmodel.layers if isinstance(layer, KerasActivationQuantizationHolder)]

    def test_quantized_model_contains_only_expected_activation_quantizers(self):
        """
        Runs PTQ and checks that the activation quantizers are the activation quantizers that we expect.
        """
        super().test_quantized_model_contains_only_expected_activation_quantizers()


    def test_quantized_model_with_reuse_contains_only_expected_activation_quantizers(self):
        """
        Runs PTQ on a model with reuse layer and checks that the activation quantizers are the activation quantizers that we expect.
        """
        super().test_quantized_model_with_reuse_contains_only_expected_activation_quantizers()

    def test_facade_ru_data_matches_expected_for_fused_graph(self):
        """
        Compute RU data on a model and check the computed max cut is as expected when we take the fusing into account.
        """
        super().test_facade_ru_data_matches_expected_for_fused_graph()

    def test_final_ru_data_is_correct(self):
        """
        Check that the activation memory in the final RU after running PTQ is as expected when we take fusing into account.
        """
        super().test_final_ru_data_is_correct()

    def test_facade_ru_data_matches_expected_with_snc_model(self):
        """
        Compute RU data on a model that goes through SNC and check the computed max cut is as expected when we take the fusing into account.
        """
        super().test_facade_ru_data_matches_expected_with_snc_model()

    def test_final_ru_data_with_snc_model(self):
        """
        Check that the activation memory in the final RU after running PTQ on a model that goes through SNC is as expected when we take fusing into account.
        """
        super().test_final_ru_data_with_snc_model()
