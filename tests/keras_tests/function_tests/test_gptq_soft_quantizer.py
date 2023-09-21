import keras
import unittest
import tensorflow as tf
from keras.models import clone_model
from tensorflow.keras.layers import Conv2D, Input
import numpy as np
import model_compression_toolkit as mct
from model_compression_toolkit.constants import THRESHOLD, MIN_THRESHOLD
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from model_compression_toolkit.core.keras.constants import KERNEL
from model_compression_toolkit.gptq.keras.quantizer.soft_rounding.symmetric_soft_quantizer import \
    SymmetricSoftRoundingGPTQ
from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerWeightsConfig
from model_compression_toolkit.trainable_infrastructure import KerasTrainableQuantizationWrapper

from tests.keras_tests.utils import get_layers_from_model_by_type

tp = mct.target_platform


def model_test(input_shape, num_channels=3, kernel_size=1):
    inputs = Input(shape=input_shape)
    outputs = Conv2D(num_channels, kernel_size, use_bias=False)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def wrap_test_model(model, per_channel=False, param_learning=False):
    tqwc = TrainableQuantizerWeightsConfig(weights_quantization_method=QuantizationMethod.SYMMETRIC,
                                           weights_n_bits=8,
                                           weights_quantization_params={THRESHOLD: 2.0},
                                           enable_weights_quantization=True,
                                           weights_channels_axis=3,
                                           weights_per_channel_threshold=per_channel,
                                           min_threshold=MIN_THRESHOLD)

    sq = SymmetricSoftRoundingGPTQ(quantization_config=tqwc,
                                   quantization_parameter_learning=param_learning)

    def _wrap(layer):
        if isinstance(layer, Conv2D):
            return KerasTrainableQuantizationWrapper(layer, weights_quantizers={'kernel': sq})
        else:
            return layer
    return clone_model(model, clone_function=_wrap)


class TestGPTQSoftQuantizer(unittest.TestCase):

    def soft_symmetric_quantizer_per_tensor(self, param_learning=False):
        input_shape = (1, 1, 1)
        in_model = model_test(input_shape)

        conv_layer = get_layers_from_model_by_type(model=in_model, layer_type=Conv2D, include_wrapped_layers=False)[0]
        t = np.asarray([0.1, -0.3, 0.67]).reshape([1] + conv_layer.weights[0].shape)
        conv_layer.set_weights(t)

        in_model = wrap_test_model(in_model, per_channel=False, param_learning=param_learning)

        input = [np.ones([1, 1, 1, 1]).astype(np.float32)]
        wrapped_conv_layer = get_layers_from_model_by_type(model=in_model, layer_type=Conv2D,
                                                           include_wrapped_layers=True)[0]
        float_weights = [x[1] for x in wrapped_conv_layer._weights_vars if x[0] == KERNEL][0]
        out = in_model(input)
        self.assertTrue(np.any(float_weights != out))

        out_t = in_model(input, training=True)
        self.assertTrue(np.all(float_weights == out_t))

    def soft_symmetric_quantizer_per_channel(self, param_learning=False):
        input_shape = (2, 2, 2)
        in_model = model_test(input_shape, num_channels=1, kernel_size=2)

        conv_layer = get_layers_from_model_by_type(model=in_model, layer_type=Conv2D, include_wrapped_layers=False)[0]
        t = np.asarray([0.1, -0.3, 0.67, -0.44, 0.51, 0.91, -0.001, 0.12])\
            .reshape([1] + conv_layer.weights[0].shape)
        conv_layer.set_weights(t)

        in_model = wrap_test_model(in_model, per_channel=True, param_learning=param_learning)

        input = [np.ones([1, 2, 2, 2]).astype(np.float32)]
        wrapped_conv_layer = get_layers_from_model_by_type(model=in_model, layer_type=Conv2D,
                                                           include_wrapped_layers=True)[0]
        float_weights = [x[1] for x in wrapped_conv_layer._weights_vars if x[0] == KERNEL][0]
        out = in_model(input)
        self.assertFalse(np.isclose(np.sum(float_weights), out))

        out_t = in_model(input, training=True)
        self.assertTrue(np.isclose(np.sum(float_weights), out_t))

    def test_soft_targets_symmetric_per_tensor(self):
        self.soft_symmetric_quantizer_per_tensor(param_learning=False)

    def test_soft_targets_symmetric_per_tensor_param_learning(self):
        self.soft_symmetric_quantizer_per_tensor(param_learning=True)

    def test_soft_targets_symmetric_per_channel(self):
        self.soft_symmetric_quantizer_per_channel(param_learning=False)

    def test_soft_targets_symmetric_per_channel_param_learning(self):
        self.soft_symmetric_quantizer_per_channel(param_learning=False)
