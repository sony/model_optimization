from tests.keras_tests.exporter_tests.tflite_int8.tflite_int8_exporter_base_test import TFLiteINT8ExporterBaseTest
import keras
import numpy as np

layers = keras.layers

class TestConv2DExporter(TFLiteINT8ExporterBaseTest):
    def get_model(self):
        return self.get_one_layer_model(layers.Conv2D(6,5))

    def run_checks(self):
        # Fetch quantized weights from int8 model tensors
        kernel_quantization_parameters, kernel_tensor_index = None, None
        for t in self.interpreter.get_tensor_details():
            if np.all(t["shape"] == np.asarray([6, 5, 5, 3])):
                kernel_tensor_index = t["index"]
                kernel_quantization_parameters = t["quantization_parameters"]
                print(kernel_quantization_parameters)
        assert kernel_quantization_parameters is not None
        assert kernel_tensor_index is not None

        # Assert there are 6 scales and zero points (like the number of output channels)
        assert len(kernel_quantization_parameters["scales"]) == 6
        assert len(kernel_quantization_parameters["zero_points"]) == 6
        assert np.all(kernel_quantization_parameters["zero_points"] == np.zeros(6))

        # Reshape Conv kernel to be at the same dimensions as in TF.
        kernel = self.interpreter.tensor(kernel_tensor_index)().transpose(1,2,3,0)
        fake_quantized_kernel_from_exportable_model = self.exportable_model.layers[2].dispatcher.weight_quantizers['kernel'](self.exportable_model.layers[2].layer.kernel)
        fake_quantized_kernel_from_int8_model = kernel * kernel_quantization_parameters["scales"].reshape(1,1,1,6)
        assert np.all(fake_quantized_kernel_from_exportable_model == fake_quantized_kernel_from_int8_model), f'Expected quantized kernel to be the same in exportable model and in int8 model'

        for tensor in self.interpreter.get_tensor_details():
            assert 'quantization_parameters' in tensor.keys()
            scales = tensor['quantization_parameters']['scales']
            assert np.all(np.log2(scales) == np.round(np.log2(scales))), f'Expected all scales to be POT but scales are {scales} in tensor {tensor["name"]}'
