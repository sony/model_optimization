from tests.keras_tests.exporter_tests.tflite_int8.tflite_int8_exporter_base_test import TFLiteINT8ExporterBaseTest
import keras
import numpy as np

layers = keras.layers

class TestDenseExporter(TFLiteINT8ExporterBaseTest):

    def get_input_shape(self):
        # More than 3 dims input to test the substitution to point-wise
        return [(4,5,6,7,8)]

    def get_model(self):
        return self.get_one_layer_model(layers.Dense(20))

    def run_checks(self):
        # assert expected output shape
        expected_output_shape = np.asarray([1, 4, 5, 6, 7, 20])
        assert np.all(self.interpreter.get_output_details()[0]['shape']==expected_output_shape), f'Expected output shape to be {expected_output_shape} but is {self.interpreter.get_output_details()[0]["shape"]}'

        # Fetch quantized weights from int8 model tensors
        kernel_quantization_parameters, kernel_tensor_index = None, None
        for t in self.interpreter.get_tensor_details():
            if np.all(t["shape"] == np.asarray([20, 1, 1, 8])):
                kernel_tensor_index = t["index"]
                kernel_quantization_parameters = t["quantization_parameters"]
        assert kernel_quantization_parameters is not None
        assert kernel_tensor_index is not None

        # Assert there are 20 scales and zero points (like the units number)
        assert len(kernel_quantization_parameters["scales"]) == 20
        assert len(kernel_quantization_parameters["zero_points"]) == 20
        assert np.all(kernel_quantization_parameters["zero_points"]==np.zeros(20))

        fake_quantized_kernel_from_exportable_model = self.exportable_model.layers[2].dispatcher.weight_quantizers['kernel'](self.exportable_model.layers[2].layer.kernel)
        # First reshape Conv kernel to be at the same dimensions as in TF.
        # Then reshape it to the original Dense kernel shape.
        # Then use scales to compute the fake quant kernel and compare it to the Dense fake quantized kernel
        kernel = self.interpreter.tensor(kernel_tensor_index)()
        fake_quantized_kernel_from_int8_model = kernel.transpose(1, 2, 3, 0).reshape(8, 20) * kernel_quantization_parameters["scales"].reshape(1, 20)
        assert np.all(fake_quantized_kernel_from_int8_model==fake_quantized_kernel_from_exportable_model), f'Expected quantized kernel to be the same in exportable model and in int8 model'

        for tensor in self.interpreter.get_tensor_details():
            assert 'quantization_parameters' in tensor.keys()
            scales = tensor['quantization_parameters']['scales']
            assert np.all(np.log2(scales) == np.round(np.log2(scales))), f'Expected all scales to be POT but scales are {scales} in tensor {tensor["name"]}'
