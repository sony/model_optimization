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
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.keras.constants import KERNEL, DEPTHWISE_KERNEL
from model_compression_toolkit.ptq import keras_post_training_quantization
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OpQuantizationConfig, \
    AttributeQuantizationConfig, Signedness
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR
from tests.common_tests.helpers.tpcs_for_tests.v4.tpc import generate_tpc

INPUT_SHAPE = (224, 224, 3)


@pytest.fixture
def rep_data_gen():
    np.random.seed(42)

    def representative_dataset():
        for _ in range(2):
            yield [np.random.randn(2, *INPUT_SHAPE)]

    return representative_dataset


def model_basic():
    inputs = layers.Input(shape=INPUT_SHAPE)
    x = layers.Conv2D(2, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=x)


def model_residual():
    inputs = layers.Input(shape=INPUT_SHAPE)
    x1 = layers.Conv2D(2, 3, padding='same')(inputs)
    x1 = layers.ReLU()(x1)

    x2 = layers.Conv2D(2, 3, padding='same')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)

    x = layers.Add()([x1, x2])

    x = layers.Flatten()(x)
    x = layers.Dense(units=10, activation='softmax')(x)

    return keras.Model(inputs=inputs, outputs=x)


def set_tpc(weights_quantization_method, per_channel):
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
def tpc_factory():
    def _tpc_factory(quant_method, per_channel):
        return set_tpc(quant_method, per_channel)
    return _tpc_factory


class TestPostTrainingQuantizationApi:
    # TODO: add tests for:
    #   1) activation only, W&A, LUT quantizer (separate)
    #   2) extend to also test with different settings features (bc, snc, etc.)
    #   3) advanced models and operators

    @pytest.mark.parametrize("quant_method", [QuantizationMethod.POWER_OF_TWO,
                                              QuantizationMethod.SYMMETRIC,
                                              QuantizationMethod.UNIFORM])
    @pytest.mark.parametrize("per_channel", [True, False])
    @pytest.mark.parametrize("model", [model_basic(), model_residual()])
    def test_ptq_pot_weights_only(self, model, rep_data_gen, tpc_factory, quant_method, per_channel):

        tpc = tpc_factory(quant_method, per_channel)
        q_model, quantization_info = keras_post_training_quantization(model, rep_data_gen,
                                                                      target_platform_capabilities=tpc)

        self._verify_quantized_model_structure(model, q_model, quantization_info)

        # Assert quantization properties
        quantized_conv_layers = [l for l in q_model.layers if isinstance(l, KerasQuantizationWrapper)]
        for quantize_wrapper in quantized_conv_layers:
            assert isinstance(quantize_wrapper.layer,
                              (layers.Conv2D, layers.DepthwiseConv2D, layers.Dense, layers.Conv2DTranspose))

            if isinstance(quantize_wrapper.layer, layers.DepthwiseConv2D):
                weights_quantizer = quantize_wrapper.weights_quantizers[DEPTHWISE_KERNEL]
                num_output_channels = (quantize_wrapper.layer.depthwise_kernel.shape[-1]
                                       * quantize_wrapper.layer.depthwise_kernel.shape[-2])
            else:
                weights_quantizer = quantize_wrapper.weights_quantizers[KERNEL]
                num_output_channels = quantize_wrapper.layer.kernel.shape[-1]

            params_shape = num_output_channels if per_channel else 1
            self._verify_weights_quantizer_params(weights_quantizer, params_shape, quant_method, per_channel)


    @staticmethod
    def _verify_weights_quantizer_params(weights_quantizer, exp_params_shape, quant_method, per_channel):
        assert weights_quantizer.per_channel == per_channel
        assert weights_quantizer.quantization_method[0] == quant_method

        if quant_method == QuantizationMethod.POWER_OF_TWO:
            assert len(weights_quantizer.threshold) == exp_params_shape
            for t in weights_quantizer.threshold:
                assert np.log2(np.abs(t)).astype(int) == np.log2(np.abs(t))
        elif quant_method == QuantizationMethod.SYMMETRIC:
            assert len(weights_quantizer.threshold) == exp_params_shape
        elif quant_method == QuantizationMethod.UNIFORM:
            assert len(weights_quantizer.min_range) == exp_params_shape
            assert len(weights_quantizer.max_range) == exp_params_shape

    @staticmethod
    def _verify_quantized_model_structure(model, q_model, quantization_info):
        assert isinstance(q_model, keras.Model)
        assert quantization_info is not None and isinstance(quantization_info, UserInformation)

        # Assert quantized model structure
        assert len([l for l in q_model.layers if isinstance(l, layers.BatchNormalization)]) == 0, \
            "Expects BN folding in quantized model."
        assert len([l for l in q_model.layers if isinstance(l, MetadataLayer)]) == 1, \
            "Expects quantized model to have a metadata stored in a dedicated layer."
        original_conv_layers = [l for l in model.layers if
                                isinstance(l, (layers.Conv2D, layers.DepthwiseConv2D, layers.Dense))]
        quantized_conv_layers = [l for l in q_model.layers if isinstance(l, KerasQuantizationWrapper)]
        assert len(original_conv_layers) > 0
        assert len(original_conv_layers) == len(quantized_conv_layers), \
            "Expects all conv layers from the original model to be wrapped with a KerasQuantizationWrapper."
