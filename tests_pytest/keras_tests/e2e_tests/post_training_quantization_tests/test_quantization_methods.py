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
import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from mct_quantizers import QuantizationMethod, KerasQuantizationWrapper
from mct_quantizers.keras.metadata import MetadataLayer
from mct_quantizers.keras.quantizers import WeightsPOTInferableQuantizer, WeightsSymmetricInferableQuantizer, \
    WeightsUniformInferableQuantizer
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.keras.constants import KERNEL
from model_compression_toolkit.ptq import keras_post_training_quantization
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OpQuantizationConfig, \
    AttributeQuantizationConfig, Signedness
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR
from tests.common_tests.helpers.tpcs_for_tests.v4.tpc import generate_tpc

INPUT_SHAPE = (24, 24, 3)
NUM_CHANNELS = {layers.Conv2D: 2, layers.Dense: 10}


def model_basic():
    inputs = layers.Input(shape=INPUT_SHAPE)
    x = layers.Conv2D(NUM_CHANNELS[layers.Conv2D], 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=x)


def model_residual():
    inputs = layers.Input(shape=INPUT_SHAPE)
    x1 = layers.Conv2D(NUM_CHANNELS[layers.Conv2D], 3, padding='same')(inputs)
    x1 = layers.ReLU()(x1)

    x2 = layers.Conv2D(NUM_CHANNELS[layers.Conv2D], 3, padding='same')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)

    x = layers.Add()([x1, x2])

    x = layers.Flatten()(x)
    x = layers.Dense(units=NUM_CHANNELS[layers.Dense], activation='softmax')(x)

    return keras.Model(inputs=inputs, outputs=x)


def _get_tpc(weights_quantization_method, per_channel):
    # TODO: currently, running E2E test with IMX500 V4 TPC from tests package
    #  we need to select a default TPC for tests, which is the one we want to verify e2e for.

    att_cfg_noquant = AttributeQuantizationConfig()
    att_cfg_quant = AttributeQuantizationConfig(weights_quantization_method=weights_quantization_method,
                                                weights_n_bits=8,
                                                weights_per_channel_threshold=per_channel,
                                                enable_weights_quantization=True)

    op_cfg = OpQuantizationConfig(default_weight_attr_config=att_cfg_quant,
                                  attr_weights_configs_mapping={KERNEL_ATTR: att_cfg_quant,
                                                                BIAS_ATTR: att_cfg_noquant},
                                  activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
                                  activation_n_bits=8,
                                  supported_input_activation_n_bits=8,
                                  enable_activation_quantization=False,  # No activation quantization
                                  quantization_preserving=False,
                                  fixed_scale=None,
                                  fixed_zero_point=None,
                                  simd_size=32,
                                  signedness=Signedness.AUTO)


    tpc = generate_tpc(default_config=op_cfg, base_config=op_cfg, mixed_precision_cfg_list=[op_cfg], name="test_tpc")

    return tpc


@pytest.fixture
def rep_data_gen():
    np.random.seed(42)

    def representative_dataset():
        for _ in range(2):
            yield [np.random.randn(2, *INPUT_SHAPE)]

    return representative_dataset


@pytest.fixture(params=[QuantizationMethod.POWER_OF_TWO,
                        QuantizationMethod.SYMMETRIC,
                        QuantizationMethod.UNIFORM])
def quant_method(request):
    return request.param


@pytest.fixture(params=[True, False])
def per_channel(request):
    return request.param


@pytest.fixture
def tpc(quant_method, per_channel):
    return _get_tpc(quant_method, per_channel)


@pytest.fixture(params=[(model_basic, {"expected_num_quantized": 1}), (model_residual, {"expected_num_quantized": 3})])
def model_scenario(request):
    return request.param


class TestPTQWithQuantizationMethods:
    # TODO: add tests for:
    #   1) activation only, W&A, LUT quantizer (separate)
    #   2) advanced models and operators

    def test_ptq_weights_only_quantization_methods(self, model_scenario, rep_data_gen, quant_method, per_channel, tpc):
        model, expected_values = model_scenario
        model = model()
        q_model, quantization_info = keras_post_training_quantization(model, rep_data_gen,
                                                                      target_platform_capabilities=tpc)

        self._verify_quantized_model_structure(q_model, quantization_info, expected_values['expected_num_quantized'])

        # Assert quantization properties
        quantized_conv_layers = [l for l in q_model.layers if isinstance(l, KerasQuantizationWrapper)]
        for quantize_wrapper in quantized_conv_layers:
            assert isinstance(quantize_wrapper.layer, (layers.Conv2D, layers.Dense))

            weights_quantizer = quantize_wrapper.weights_quantizers[KERNEL]
            num_output_channels = NUM_CHANNELS[type(quantize_wrapper.layer)]

            params_shape = num_output_channels if per_channel else 1
            self._verify_weights_quantizer_params(weights_quantizer, params_shape, quant_method, per_channel)


    @staticmethod
    def _verify_weights_quantizer_params(weights_quantizer, exp_params_shape, quant_method, per_channel):
        assert weights_quantizer.per_channel == per_channel
        assert weights_quantizer.quantization_method[0] == quant_method

        if quant_method == QuantizationMethod.POWER_OF_TWO:
            assert isinstance(weights_quantizer, WeightsPOTInferableQuantizer)
            assert len(weights_quantizer.threshold) == exp_params_shape
            for t in weights_quantizer.threshold:
                assert np.log2(np.abs(t)).astype(int) == np.log2(np.abs(t))
        elif quant_method == QuantizationMethod.SYMMETRIC:
            assert isinstance(weights_quantizer, WeightsSymmetricInferableQuantizer)
            assert len(weights_quantizer.threshold) == exp_params_shape
        elif quant_method == QuantizationMethod.UNIFORM:
            assert isinstance(weights_quantizer, WeightsUniformInferableQuantizer)
            assert len(weights_quantizer.min_range) == exp_params_shape
            assert len(weights_quantizer.max_range) == exp_params_shape

    @staticmethod
    def _verify_quantized_model_structure(q_model, quantization_info, expected_num_quantized):
        assert isinstance(q_model, keras.Model)
        assert quantization_info is not None and isinstance(quantization_info, UserInformation)

        # Assert quantized model structure
        assert len([l for l in q_model.layers if isinstance(l, layers.BatchNormalization)]) == 0, \
            "Expects BN folding in quantized model."
        assert len([l for l in q_model.layers if isinstance(l, MetadataLayer)]) == 1, \
            "Expects quantized model to have a metadata stored in a dedicated layer."
        quantized_layers = [l for l in q_model.layers if isinstance(l, KerasQuantizationWrapper)]
        assert len(quantized_layers) == expected_num_quantized, \
            "Expects all conv layers from the original model to be wrapped with a KerasQuantizationWrapper."
