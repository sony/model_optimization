# FAQ

## 1. Quantized model size is the same as the original model size

MCT performs a process known as *fake quantization*, wherein the model's weights and activations are still represented in a floating-point
format but are quantized to represent a maximum of 2^N unique values (for N-bit cases).

To truly benefit from quantization in terms of model size, users are encouraged to export their models to the INT8 format (currently, this is supported only for Keras models when they are exported to TFLite models).

For more information and an implementation example, check out the [INT8 TFLite export tutorial](https://github.com/sony/model_optimization/blob/main/tutorials/notebooks/keras/export/example_keras_export.ipynb)
