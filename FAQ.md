# FAQ

**Table of Contents:**

1. [Why does the size of the quantized model remain the same as the original model size?](#1-why-does-the-size-of-the-quantized-model-remain-the-same-as-the-original-model-size)
2. [Why does loading a quantized exported model from a file fail?](#2-why-does-loading-a-quantized-exported-model-from-a-file-fail)
3. [Why am I getting a torch.fx error?](#3-why-am-i-getting-a-torchfx-error)


### 1. Why does the size of the quantized model remain the same as the original model size?

MCT performs a process known as *fake quantization*, wherein the model's weights and activations are still represented in a floating-point
format but are quantized to represent a maximum of 2^N unique values (for N-bit cases).

Exporting your model to INT8 format (currently, this is supported only for Keras models exported to TFLite models) will truly compress your model,
but this exporting method is limited to uniform 8-bit quantization only.  
Note that the IMX500 converter accepts the "fake quantization" model and supports all the features of MCT (e.g. less than 8 bits for weights bit-width and non-uniform quantization).
 
For more information and an implementation example, check out the [INT8 TFLite export tutorial](https://github.com/sony/model_optimization/blob/main/tutorials/notebooks/keras/export/example_keras_export.ipynb)


### 2. Why does loading a quantized exported model from a file fail?

The models MCT exports contain QuantizationWrappers and Quantizer objects that define and quantize the model at inference.
These objects are custom layers and layer wrappers created by MCT (defined in an external repository: [MCTQ](https://github.com/sony/mct_quantizers)), 
and thus, MCT offers an API for loading these models from a file, depending on the framework.

#### Keras

Keras models can be loaded with the following function:
```python
import model_compression_toolkit as mct

quantized_model = mct.keras_load_quantized_model('my_model.keras')
```

#### PyTorch

PyTorch models can be exported as onnx models. An example of loading a saved onnx model can be found [here](https://github.com/sony/model_optimization/blob/main/tutorials/notebooks/pytorch/export/example_pytorch_export.ipynb).

*Note:* Running inference on an ONNX model in the `onnxruntime` package has a high latency.
Inference on the target platform (e.g. the IMX500) is not affected by this latency.


### 3. Why am I getting a torch.fx error?

When quantizing a PyTorch model, MCT's initial step involves converting the model into a graph representation using `torch.fx`.
However, `torch.fx` comes with certain common limitations, with the primary one being its requirement for the computational graph to remain static.

Despite these limitations, some adjustments can be made to facilitate MCT quantization.

**Solution**: (assuming you have access to the model's code)

Check the `torch.fx` error, and search for an identical replacement. Some examples:
* An `if` statement in a module's `forward` method might can be easily skipped.
* The `list()` Python method can be replaced with a concatenation operation [A, B, C].
