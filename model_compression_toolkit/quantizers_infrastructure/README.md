# Quantizers Infrastructure (QI)

Quantizers infrastructure is a module containing quantization abstraction and quantizers for hardware-oriented model optimization tools such as the Model Compression Toolkit ([MCT](https://github.com/sony/model_optimization)).

It provides the required abstraction for emulating inference-time quantization and optimization methods such as quantization aware training. 

## High level description

To create quantization abstraction in a model,
we replace each layer with a "Quantization Wrapper" - a more complex layer, which includes the original layer, a weights quantizer block, and activations quantizer block. A quantization dispatcher is attached to each quantization wrapper to set the quantizer type.

<img src="../../docsrc/images/quantization_infra.png" width="700">

The quantizers in this module are divided into two main types:
"Inferable Quantizer", is used for emulating inference-time quantization, and "Trainable Quantizer", contains learnable quantization parameters that can be optimized during training.

More details and "how to" examples for Tensorflow can be found in:

[Inferable quantizers for Tensorflow/Keras](keras/inferable_quantizers/README.md)

[Trainable quantizers for Tensorflow/Keras](keras/README.md)

And for pytorch:

[Inferable quantizers for pytorch](pytorch/inferable_quantizers/README.md)

[Trainable quantizers for pytorch](pytorch/README.md)

  



