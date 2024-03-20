# FAQ

## 1. Quantized model size is the same as the original model size

MCT performs a process known as *fake quantization*, wherein the model's weights and activations are still represented in a floating-point
format but are quantized to represent a maximum of 2^N unique values (for N-bit cases).

Exporting your model to INT8 format (currently, this is supported only for Keras models exported to TFLite models) will truly compress your model,
but this exporting method is limited to uniform 8 bit quantization only.  
Note that the IMX500 converter accepts the "fake quantization" model and supports all the features of MCT (e.g. less than 8 bits for weights bit-width and non-uniform quantization).
 
For more information and an implementation example, check out the [INT8 TFLite export tutorial](https://github.com/sony/model_optimization/blob/main/tutorials/notebooks/keras/export/example_keras_export.ipynb)


## 2. Loading exported models

The models MCT exports contain QuantizationWrappers and Quantizer object that define and quantize the model at inference.
These objects are custom layers created by MCT, so MCT offers a simple API for loading these models from file, depending on the framework.

### Keras

Keras models can be loaded with the following function:
```python
import model_compression_toolkit as mct

quantized_model = mct.keras_load_quantized_model('my_model.keras')
```

### PyTorch

PyTorch models can be exported as onnx models. An example of loading a saved onnx model can be found [here](https://sony.github.io/model_optimization/api/api_docs/modules/exporter.html#use-exported-model-for-inference).

