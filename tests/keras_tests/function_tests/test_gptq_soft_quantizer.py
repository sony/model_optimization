import keras
import unittest

from keras.models import clone_model
from tensorflow.keras.layers import Conv2D, Input
import numpy as np
import model_compression_toolkit as mct
from model_compression_toolkit.core.common.constants import THRESHOLD
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from model_compression_toolkit.core.keras.constants import KERNEL
from model_compression_toolkit.gptq.keras.quantizer.soft_rounding.symmetric_soft_quantizer import \
    SymmetricSoftRoundingGPTQ
from model_compression_toolkit.quantizers_infrastructure import TrainableQuantizerWeightsConfig, \
    KerasQuantizationWrapper

tp = mct.target_platform


def model_test(input_shape, per_channel, num_channels=3, kernel_size=1):
    inputs = Input(shape=input_shape)
    outputs = Conv2D(num_channels, kernel_size, use_bias=False)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return wrap_test_model(model, per_channel)


def wrap_test_model(model, per_channel=False):
    tqwc = TrainableQuantizerWeightsConfig(weights_quantization_method=QuantizationMethod.SYMMETRIC,
                                           weights_n_bits=8,
                                           weights_quantization_params={THRESHOLD: 2.0},
                                           enable_weights_quantization=True,
                                           weights_channels_axis=3,
                                           weights_per_channel_threshold=per_channel,
                                           min_threshold=0)

    sq = SymmetricSoftRoundingGPTQ(quantization_config=tqwc,
                                   quantization_parameter_learning=False)

    def _wrap(layer):
        if isinstance(layer, Conv2D):
            return KerasQuantizationWrapper(layer, weights_quantizers={'kernel': sq})
        else:
            return layer
    return clone_model(model, clone_function=_wrap)


class TestGPTQSoftQuantizer(unittest.TestCase):

    def test_soft_targets_symmetric_per_tensor(self):
        input_shape = (1, 1, 1)
        in_model = model_test(input_shape, per_channel=False)

        input = [np.ones([1, 1, 1, 1]).astype(np.float32)]
        float_weights = [x[1] for x in in_model.layers[1]._weights_vars if x[0] == KERNEL][0]
        out = in_model(input)
        self.assertTrue(np.any(float_weights != out))

        out_t = in_model(input, training=True)
        self.assertTrue(np.all(float_weights == out_t))

    def test_soft_targets_symmetric_per_channel(self):
        input_shape = (2, 2, 2)
        in_model = model_test(input_shape, per_channel=True, num_channels=1, kernel_size=2)

        input = [np.ones([1, 2, 2, 2]).astype(np.float32)]
        float_weights = [x[1] for x in in_model.layers[1]._weights_vars if x[0] == KERNEL][0]
        out = in_model(input)
        self.assertFalse(np.isclose(np.sum(float_weights), out))

        out_t = in_model(input, training=True)
        self.assertTrue(np.isclose(np.sum(float_weights), out_t))
