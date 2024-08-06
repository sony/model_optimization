from typing import List, Any, Tuple
from functools import partial

import tensorflow as tf
from mct_quantizers import KerasActivationQuantizationHolder

from model_compression_toolkit.trainable_infrastructure import KerasTrainableQuantizationWrapper
from model_compression_toolkit.ptq import keras_post_training_quantization
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.keras_tests.tpc_keras import get_quantization_disabled_keras_tpc
from packaging import version
if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src import Input, Model
    from keras.src.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose, Reshape, ZeroPadding2D, \
        Dropout, MaxPooling2D, Activation, ReLU, GlobalAveragePooling2D, Add, Multiply, AveragePooling2D, \
        UpSampling2D, InputLayer, Concatenate, Softmax, PReLU, Flatten, Cropping2D, Dot, ELU, LeakyReLU, Permute, \
        LayerNormalization
else:
    from keras import Input, Model
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose, Reshape, ZeroPadding2D, \
        Dropout, MaxPooling2D, Activation, ReLU, GlobalAveragePooling2D, Add, Multiply, AveragePooling2D, \
        UpSampling2D, InputLayer, Concatenate, Softmax, PReLU, Flatten, Cropping2D, Dot, ELU, LeakyReLU, Permute, \
        LayerNormalization

from model_compression_toolkit.core import FrameworkInfo
from model_compression_toolkit.gptq import keras_gradient_post_training_quantization
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from tests.common_tests.base_layer_test import BaseLayerTest, LayerTestMode
import numpy as np


KERAS_LAYER_TEST_OPS = {
    "kernel_ops": [Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose],

    "no_quantization": [Reshape, tf.reshape, Flatten, Permute, Cropping2D, ZeroPadding2D, Dropout, MaxPooling2D,
                        tf.reshape, tf.split, tf.quantization.fake_quant_with_min_max_vars],

    "activation": [Activation, ReLU, tf.nn.relu, tf.nn.relu6, tf.nn.leaky_relu, Softmax, GlobalAveragePooling2D, Add,
                   Multiply, AveragePooling2D, UpSampling2D, InputLayer, Concatenate, PReLU, ELU, tf.nn.silu,
                   tf.nn.swish, tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu, tf.nn.relu6, tf.nn.leaky_relu, LeakyReLU,
                   tf.nn.softsign, tf.nn.gelu, tf.nn.elu, tf.nn.selu, tf.nn.softplus, tf.nn.softmax, Dot,
                   LayerNormalization, tf.add, tf.multiply, tf.reduce_mean, tf.reduce_min, tf.reduce_sum, tf.reduce_max,
                   tf.image.resize, tf.image.crop_and_resize, tf.concat,
                   ]
}


class BaseKerasLayerTest(BaseLayerTest):
    def __init__(self,
                 unit_test,
                 layers: List[Any],
                 val_batch_size: int = 1,
                 num_calibration_iter: int = 1,
                 num_of_inputs: int = 1,
                 input_shape: Tuple[int, int, int] = (8, 8, 3),
                 quantization_modes: List[LayerTestMode] = [LayerTestMode.FLOAT, LayerTestMode.QUANTIZED_8_BITS],
                 is_inputs_a_list: bool = False,
                 use_cpu: bool = False):

        super().__init__(unit_test=unit_test,
                         layers=layers,
                         val_batch_size=val_batch_size,
                         num_calibration_iter=num_calibration_iter,
                         num_of_inputs=num_of_inputs,
                         input_shape=input_shape,
                         quantization_modes=quantization_modes,
                         is_inputs_a_list=is_inputs_a_list,
                         use_cpu=use_cpu)

    def get_tpc(self):
        if self.current_mode == LayerTestMode.FLOAT:
            # Disable all features that are enabled by default:
            return get_quantization_disabled_keras_tpc("float_layer_test")
        elif self.current_mode == LayerTestMode.QUANTIZED_8_BITS:
            tp = generate_test_tp_model({'weights_n_bits': 8,
                                         'activation_n_bits': 8})
            return generate_keras_tpc(name="8bit_layer_test", tp_model=tp)
        else:
            raise NotImplemented

    def get_fw_info(self) -> FrameworkInfo:
        return DEFAULT_KERAS_INFO

    def get_fw_impl(self) -> FrameworkImplementation:
        return KerasImplementation()

    def get_ptq_facade(self):
        return keras_post_training_quantization

    def get_gptq_facade(self):
        return keras_gradient_post_training_quantization

    def predict(self, model: Model, input: List[np.ndarray]):
        if self.use_cpu:
            with tf.device('/cpu:0'):
                return model.predict(input)
        return model.predict(input)

    def create_networks(self):
        layers = self.get_layers()
        networks = []
        for i, layer in enumerate(layers):
            inputs = [Input(shape=s[1:]) for s in self.get_input_shapes()]
            if isinstance(layer, partial) and layer.func is tf.image.combined_non_max_suppression:
                # need to identify the tf.image.combined_non_max_suppression because it has 2 separate inputs
                outputs = layer(inputs[0], tf.reduce_mean(inputs[0], 3))
            elif self.is_inputs_a_list:
                outputs = layer(inputs)
            else:
                outputs = layer(*inputs)
            m = Model(inputs=inputs, outputs=outputs)
            networks.append(m)
        return networks

    def compare(self, quantized_model: Model, float_model: Model, input_x=None, quantization_info=None):
        # Assert things that should happen when using FLOAT quantization mode
        if self.current_mode == LayerTestMode.FLOAT:
            self.__compare_float_mode(float_model, quantized_model)

        # Assert things that should happen when using QUANTIZED_8_BITS quantization mode
        elif self.current_mode == LayerTestMode.QUANTIZED_8_BITS:
            self.__compare_8bits_quantization_mode(float_model, quantized_model)

        ####################################################################
        # Assert conditions that should be valid for ALL quantization modes
        ###################################################################
        self.unit_test.assertTrue(len(quantized_model.outputs) == len(float_model.outputs))
        self.unit_test.assertTrue(len(quantized_model.inputs) == len(float_model.inputs))

        # Check inference is possible
        self.predict(quantized_model, self.generate_inputs())

        # Check equal output shapes for both models:
        for quantized_model_output, float_model_output in zip(quantized_model.outputs, float_model.outputs):
            self.unit_test.assertTrue(quantized_model_output.shape.as_list() == float_model_output.shape.as_list())

    def __compare_8bits_quantization_mode(self, float_model, quantized_model):
        fw_info = self.get_fw_info()
        for layer in quantized_model.layers:
            if not isinstance(layer, InputLayer):
                if isinstance(layer, KerasActivationQuantizationHolder):
                    assert layer.activation_holder_quantizer.get_config()['num_bits'] == 8
                if isinstance(layer, KerasTrainableQuantizationWrapper):
                    # Assert the kernel op outputs are quantized
                    assert isinstance(layer.outbound_nodes[0].layer, KerasActivationQuantizationHolder)
                    assert len(layer.weights_quantizers) > 0
                    for q in layer.weights_quantizers.values():
                        assert q.get_config()['num_bits'] == 8
                    for attr in fw_info.get_kernel_op_attributes(type(layer.layer)):
                        self.unit_test.assertTrue(np.sum(np.abs(
                            layer.get_quantized_weights()[attr] - getattr(float_model.get_layer(layer.layer.name), attr))) > 0.0)

    def __compare_float_mode(self, float_model, quantized_model):
        for layer_index, layer in enumerate(quantized_model.layers):
            # Check there are no fake-quant layers
            self.unit_test.assertFalse(isinstance(layer, KerasActivationQuantizationHolder))
            if not isinstance(layer, InputLayer):
                if isinstance(layer, KerasTrainableQuantizationWrapper):
                    assert len(layer.activation_quantizers) == 0
                    assert len(layer.weights_quantizers.keys()) == 0
                    # check unchanged weights
                    if hasattr(layer.layer, 'weights') and len(layer.layer.weights) > 0:
                        for i, w in enumerate(layer.layer.weights):
                            self.unit_test.assertTrue(np.sum(np.abs(w - float_model.layers[layer_index-1].weights[i])) == 0.0)

        input_tensors = self.generate_inputs()
        y = self.predict(float_model, input_tensors)
        y_hat = self.predict(quantized_model, input_tensors)
        if isinstance(y, list) or (isinstance(self.layers[0], partial) and
                                   self.layers[0].func is tf.image.combined_non_max_suppression):
            # the combined_non_max_suppression is wrapped in "partial"
            # the output of tf.image.combined_non_max_suppression is not a list but can be handled the same
            for fo, qo in zip(y, y_hat):
                distance = np.sum(np.abs(fo - qo))
                self.unit_test.assertTrue(distance == 0,
                                          msg=f'Outputs should be identical. Observed distance: {distance}')
        else:
            distance = np.sum(np.abs(y - y_hat))
            self.unit_test.assertTrue(distance == 0,
                                      msg=f'Outputs should be identical. Observed distance: {distance}')
