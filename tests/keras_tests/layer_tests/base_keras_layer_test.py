from typing import List, Any, Tuple

import tensorflow as tf

if tf.__version__ < "2.6":
    from tensorflow.python.keras.layers.core import TFOpLambda
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
else:
    from keras.layers.core import TFOpLambda
    from keras import Input, Model

from model_compression_toolkit import FrameworkInfo, keras_post_training_quantization, \
    keras_post_training_quantization_mixed_precision
from model_compression_toolkit.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.keras.back2framework.model_builder import is_layer_fake_quant
from model_compression_toolkit.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.keras.keras_implementation import KerasImplementation
from tests.common_tests.base_layer_test import BaseLayerTest, LayerTestMode
import numpy as np


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

    def get_fw_info(self) -> FrameworkInfo:
        return DEFAULT_KERAS_INFO

    def get_fw_impl(self) -> FrameworkImplementation:
        return KerasImplementation()

    def get_ptq_facade(self):
        return keras_post_training_quantization

    def get_mixed_precision_ptq_facade(self):
        return keras_post_training_quantization_mixed_precision

    # def predict(self, model: Model, input: List[np.ndarray]):
    #     if self.use_cpu:
    #         with tf.device('/cpu:0'):
    #             return model.predict(input)
    #     return model.predict(input)

    def create_networks(self):
        layers = self.get_layers()
        networks = []
        for i, layer in enumerate(layers):
            inputs = [Input(shape=s[1:]) for s in self.get_input_shapes()]
            if self.is_inputs_a_list:
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
        ####################################################################
        self.unit_test.assertTrue(len(quantized_model.outputs) == len(float_model.outputs))
        self.unit_test.assertTrue(len(quantized_model.inputs) == len(float_model.inputs))

        # Check inference is possible
        # self.predict(quantized_model, self.generate_inputs())
        quantized_model.predict(self.generate_inputs())

        # Check equal output shapes for both models:
        for quantized_model_output, float_model_output in zip(quantized_model.outputs, float_model.outputs):
            self.unit_test.assertTrue(quantized_model_output.shape.as_list() == float_model_output.shape.as_list())

    def __compare_8bits_quantization_mode(self, float_model, quantized_model):
        fw_info = self.get_fw_info()
        for layer in quantized_model.layers:
            op = layer.function if isinstance(layer, TFOpLambda) else type(layer)
            if op in fw_info.kernel_ops:
                for attr in fw_info.get_kernel_op_attributes(type(layer)):
                    self.unit_test.assertTrue(np.sum(np.abs(
                        getattr(layer, attr) - getattr(float_model.get_layer(layer.name), attr))) > 0.0)
                for next_layer in [node.layer for node in layer.outbound_nodes]:
                    self.unit_test.assertTrue(is_layer_fake_quant(next_layer))

            elif op in fw_info.activation_ops:
                for next_layer in [node.layer for node in layer.outbound_nodes]:
                    self.unit_test.assertTrue(is_layer_fake_quant(next_layer))

            elif op in fw_info.no_quantization_ops:
                for next_layer in [node.layer for node in layer.outbound_nodes]:
                    self.unit_test.assertFalse(is_layer_fake_quant(next_layer))

            else:
                raise Exception('Layer is not in framework info')

    def __compare_float_mode(self, float_model, quantized_model):
        for layer_index, layer in enumerate(quantized_model.layers):
            # Check there are no fake-quant layers
            self.unit_test.assertFalse(is_layer_fake_quant(layer))
            # check unchanged weights
            if hasattr(layer, 'weights') and len(layer.weights) > 0:
                for i, w in enumerate(layer.weights):
                    self.unit_test.assertTrue(np.sum(np.abs(w - float_model.layers[layer_index].weights[i])) == 0.0)

            input_tensors = self.generate_inputs()
            # y = self.predict(float_model, input_tensors)
            y = float_model.predict(input_tensors)
            # y_hat = self.predict(quantized_model, input_tensors)
            y_hat = quantized_model.predict(input_tensors)
            if isinstance(y, list):
                for fo, qo in zip(y, y_hat):
                    distance = np.sum(np.abs(fo - qo))
                    self.unit_test.assertTrue(distance == 0,
                                              msg=f'Outputs should be identical. Observed distance: {distance}')

            else:
                distance = np.sum(np.abs(y - y_hat))
                self.unit_test.assertTrue(distance == 0,
                                          msg=f'Outputs should be identical. Observed distance: {distance}')
