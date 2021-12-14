from typing import List

import tensorflow as tf

if tf.__version__ < "2.6":
    from tensorflow.python.keras.layers.core import TFOpLambda
else:
    from keras.layers.core import TFOpLambda
    from keras import Input, Model

from model_compression_toolkit import FrameworkInfo, keras_post_training_quantization, \
    keras_post_training_quantization_mixed_precision, MixedPrecisionQuantizationConfig
from model_compression_toolkit.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.keras.back2framework.model_builder import is_layer_fake_quant
from model_compression_toolkit.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.keras.keras_implementation import KerasImplementation
from tests.common_tests.base_layer_test import BaseLayerTest
from tests.common_tests.base_test import TestMode
import numpy as np


class BaseKerasLayerTest(BaseLayerTest):
    def __init__(self,
                 unit_test,
                 val_batch_size=1,
                 num_calibration_iter=1,
                 num_of_inputs=1,
                 input_shape=(8, 8, 3),
                 quantization_modes: List[TestMode] = [TestMode.FLOAT, TestMode.QUANTIZED_16_BITS],
                 is_inputs_a_list=False):

        super().__init__(unit_test=unit_test,
                         val_batch_size=val_batch_size,
                         num_calibration_iter=num_calibration_iter,
                         num_of_inputs=num_of_inputs,
                         input_shape=input_shape,
                         quantization_modes=quantization_modes,
                         is_inputs_a_list=is_inputs_a_list)

    def get_layers(self):
        raise Exception('Implement get_layers')

    def get_fw_info(self) -> FrameworkInfo:
        return DEFAULT_KERAS_INFO

    def get_fw_impl(self) -> FrameworkImplementation:
        return KerasImplementation()

    def get_ptq_facade(self):
        return keras_post_training_quantization

    def get_mixed_precision_ptq_facade(self):
        return keras_post_training_quantization_mixed_precision

    def create_networks(self):
        layers = self.get_layers()
        networks = []
        for i, layer in enumerate(layers):
            print(f'Test layer {i}: {self.__class__.__name__}')
            inputs = [Input(shape=s[1:]) for s in self.get_input_shapes()]
            if self.is_inputs_a_list:
                outputs = layer(inputs)
            else:
                outputs = layer(*inputs)
            m = Model(inputs=inputs, outputs=outputs)
            networks.append(m)
        return networks

    def run_test(self):
        x = self.generate_inputs()

        def representative_data_gen():
            return x

        feature_networks = self.create_networks()
        feature_networks = feature_networks if isinstance(feature_networks, list) else [feature_networks]
        for model_float in feature_networks:
            for mode in self.quantization_modes:
                self.quantization_mode = mode
                print(f'Mode: {self.quantization_mode}')
                qc = self.get_quantization_config()
                if isinstance(qc, MixedPrecisionQuantizationConfig):
                    ptq_model, quantization_info = self.get_mixed_precision_ptq_facade()(model_float,
                                                                                         representative_data_gen,
                                                                                         n_iter=self.num_calibration_iter,
                                                                                         quant_config=qc,
                                                                                         fw_info=self.get_fw_info())
                else:
                    ptq_model, quantization_info = self.get_ptq_facade()(model_float,
                                                                         representative_data_gen,
                                                                         n_iter=self.num_calibration_iter,
                                                                         quant_config=qc,
                                                                         fw_info=self.get_fw_info())

                self.compare(ptq_model, model_float, input_x=x, quantization_info=quantization_info)

    def compare(self, quantized_model: Model, float_model: Model, input_x=None, quantization_info=None):
        # Assert things that should happen when using FLOAT quantization mode
        if self.quantization_mode == TestMode.FLOAT:
            self.__compare_float_mode(float_model, quantized_model)

        # Assert things that should happen when using QUANTIZED_16_BITS quantization mode
        elif self.quantization_mode == TestMode.QUANTIZED_16_BITS:
            self.__compare_16bits_quantization_mode(float_model, quantized_model)

        ####################################################################
        # Assert conditions that should be valid for ALL quantization modes
        ####################################################################
        self.unit_test.assertTrue(len(quantized_model.outputs) == len(float_model.outputs))
        self.unit_test.assertTrue(len(quantized_model.inputs) == len(float_model.inputs))

        # Check inference and equal output shapes for both models:
        predictions = quantized_model.predict(self.generate_inputs())
        predictions = predictions if isinstance(predictions, list) else [predictions]
        for prediction, float_model_output in zip(predictions, float_model.outputs):
            self.unit_test.assertTrue(prediction.shape[1:] == float_model_output.shape[1:])  # ignore batch dimension

    def __compare_16bits_quantization_mode(self, float_model, quantized_model):
        fw_info = self.get_fw_info()
        for layer in quantized_model.layers:
            op = layer.function if isinstance(layer, TFOpLambda) else type(layer)
            if op in fw_info.kernel_ops:
                for attr in fw_info.get_kernel_op_attributes(type(layer)):
                    self.unit_test.assertTrue(np.sum(np.abs(
                        getattr(layer, attr) - getattr(float_model.get_layer(layer.name), attr))) > 0.0)
                for next_layer in [node.layer for node in layer.outbound_nodes]:
                    self.unit_test.assertTrue(is_layer_fake_quant(next_layer))

            if op in fw_info.activation_ops:
                for next_layer in [node.layer for node in layer.outbound_nodes]:
                    self.unit_test.assertTrue(is_layer_fake_quant(next_layer))

            if op in fw_info.no_quantization_ops:
                for next_layer in [node.layer for node in layer.outbound_nodes]:
                    self.unit_test.assertFalse(is_layer_fake_quant(next_layer))

    def __compare_float_mode(self, float_model, quantized_model):
        for layer_index, layer in enumerate(quantized_model.layers):
            # Check there are no fake-quant layers
            self.unit_test.assertFalse(is_layer_fake_quant(layer))
            # check unchanged weights
            if hasattr(layer, 'weights') and len(layer.weights) > 0:
                for i, w in enumerate(layer.weights):
                    self.unit_test.assertTrue(np.sum(np.abs(w - float_model.layers[layer_index].weights[i])) == 0.0)

            input_tensors = self.generate_inputs()
            y = float_model.predict(input_tensors)
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
