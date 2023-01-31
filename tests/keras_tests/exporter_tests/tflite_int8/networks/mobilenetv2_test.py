from keras.applications import MobileNetV2
import numpy as np

from tests.keras_tests.exporter_tests.tflite_int8.tflite_int8_exporter_base_test import TFLiteINT8ExporterBaseTest
import keras
layers = keras.layers

class TestMBV2Exporter(TFLiteINT8ExporterBaseTest):

    def get_input_shape(self):
        return [(224,224,3)]
    
    def get_model(self):
        return MobileNetV2()

    def run_checks(self):
        for tensor in self.interpreter.get_tensor_details():
            assert 'quantization_parameters' in tensor.keys()
            scales = tensor['quantization_parameters']['scales']
            assert np.all(np.log2(scales) == np.round(np.log2(scales))), f'Expected all scales to be POT but scales are {scales} in tensor {tensor["name"]}'

#
# class TestMefficientnetExporter(TestMBV2Exporter):
#     def get_input_shape(self):
#         return [(280,280,3)]
#
#     def get_model(self):
#         return MEfficientNet()
