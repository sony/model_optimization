# Quantizers Infrastructure (QI)

Quantizers infrastructure is a module containing quantization abstraction and quantizers for hardware-oriented model optimization tools such as the Model Compression Toolkit ([MCT](https://github.com/sony/model_optimization)).

It provides the required abstraction for emulating inference-time quantization and trainable quantization methods such as quantization-aware training.

## High level description

For each layer, we wrap the layer and a quantization dispatcher (which contains the quantizers and all quantization information we need to quantize the layer) in a "Quantization Wrapper".

Notice that the quantization wrapper and the quantization dispatcher are per framework.




<img src="../../docsrc/images/quantization_infra.png" width="700">

The quantizers in this module are divided into two main types:
The "Inferable Quantizer" is used for emulating inference-time quantization, and the "Trainable Quantizer", contains learnable quantization parameters that can be optimized during training.

## Details and Examples

More details and "how to" examples for Tensorflow can be found in:

[Inferable quantizers for TensorFlow](keras/inferable_quantizers/README.md)

[Trainable quantizers for TensorFlow](keras/README.md)

And for pytorch:

[Inferable quantizers for PyTorch](pytorch/inferable_quantizers/README.md)

[Trainable quantizers for PyTorch](pytorch/README.md)

  



