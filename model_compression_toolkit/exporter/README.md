## Introduction

Export your quantized model in the following serialization formats:

* TensorFlow models can be exported as Tensorflow models (`.keras` extensions) and TFLite models (`.tflite` extension).
* PyTorch models can be exported as torch script models and ONNX models (`.onnx` extension).

You can export your quantized model in the following quantization formats:
* FAKELY_QUANT: Weights and activations values are quantized but represented in float32 dtype. In this format we use the framework's quantizers. This is called fake since the values are still in floating point.
* INT8: Where weights and activations are represented using 8bits integers.
* MCTQ: Weights and activations values are quantized but represented in float32 dtype. In this format we use custom quantizer layers - [mct_quantizers](https://github.com/sony/mct_quantizers#readme).



## Usage Examples

Try our notebooks to export [Keras](../../tutorials/notebooks/mct_features_notebooks/keras/example_keras_export.ipynb) or [Pytorch](../../tutorials/notebooks/mct_features_notebooks/pytorch/example_pytorch_export.ipynb) models using different formats.
