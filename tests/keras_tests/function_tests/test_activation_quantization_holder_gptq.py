import unittest

import keras
import numpy as np
from keras.layers import ReLU
from mct_quantizers import KerasActivationQuantizationHolder, KerasQuantizationWrapper
from tensorflow.keras.layers import Conv2D, Input

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.mixed_precision.bit_width_setter import set_bit_widths
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.gptq.keras.gptq_keras_implementation import GPTQKerasImplemantation
from model_compression_toolkit.gptq.keras.gptq_training import KerasGPTQTrainer
from model_compression_toolkit.target_platform_capabilities.tpc_models.default_tpc.latest import generate_keras_tpc
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_quantization_parameters

tp = mct.target_platform


def basic_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(3, 3)(inputs)
    x = Conv2D(3, 3)(x)
    return keras.Model(inputs=inputs, outputs=x)

def activation_quantization_for_relu_model(input_shape):
    # Check that activation holder was added to GPTQ after ReLU
    inputs = Input(shape=input_shape)
    x = Conv2D(3, 3)(inputs)
    x = ReLU()(x)
    x = Conv2D(3, 3)(x)
    return keras.Model(inputs=inputs, outputs=x)

def reuse_model(input_shape):
    conv = Conv2D(3, 3)
    inputs = Input(shape=input_shape)
    x = conv(inputs)
    x = conv(x)
    x = Conv2D(3, 3)(x)
    return keras.Model(inputs=inputs, outputs=x)


def representative_dataset():
    yield [np.random.randn(1, 8, 8, 3).astype(np.float32)]


class TestGPTQModelBuilderWithActivationHolder(unittest.TestCase):

    def test_adding_holder_instead_quantize_wrapper(self):
        input_shape = (8, 8, 3)
        gptq_model = self._get_gptq_model(input_shape, basic_model)
        self.assertTrue(isinstance(gptq_model.layers[3], KerasActivationQuantizationHolder))

    def test_adding_holders_after_reuse(self):
        input_shape = (8, 8, 3)
        gptq_model = self._get_gptq_model(input_shape, reuse_model)
        self.assertTrue(isinstance(gptq_model.layers[3], KerasActivationQuantizationHolder))
        self.assertTrue(isinstance(gptq_model.layers[4], KerasActivationQuantizationHolder))

        # Test that two holders are getting inputs from reused conv2d (the layer that is wrapped)
        self.assertTrue(gptq_model.layers[2].get_output_at(0).ref() == gptq_model.layers[3].input.ref())
        self.assertTrue(gptq_model.layers[2].get_output_at(1).ref() == gptq_model.layers[4].input.ref())

    def test_adding_holder_after_relu(self):
        input_shape = (8, 8, 3)
        gptq_model = self._get_gptq_model(input_shape, activation_quantization_for_relu_model)
        self.assertTrue(isinstance(gptq_model.layers[3], ReLU))
        self.assertTrue(isinstance(gptq_model.layers[4], KerasActivationQuantizationHolder))


    def _get_gptq_model(self, input_shape, get_model_fn):
        in_model = get_model_fn(input_shape)
        keras_impl = GPTQKerasImplemantation()
        graph = prepare_graph_with_quantization_parameters(in_model,
                                                           keras_impl,
                                                           DEFAULT_KERAS_INFO,
                                                           representative_dataset,
                                                           generate_keras_tpc,
                                                           (1,) + input_shape,
                                                           mixed_precision_enabled=False)
        graph = set_bit_widths(mixed_precision_enable=False,
                               graph=graph)
        trainer = KerasGPTQTrainer(graph,
                                   graph,
                                   mct.gptq.get_keras_gptq_config(1, use_hessian_based_weights=False),
                                   keras_impl,
                                   DEFAULT_KERAS_INFO,
                                   representative_dataset)
        gptq_model, _ = trainer.build_gptq_model()
        return gptq_model


