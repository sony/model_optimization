## Introduction

[`BaseKerasTrainableQuantizer`](base_keras_quantizer.py) is an interface that enables easy quantizers development and training. 
Using this base class makes it simple to implement new quantizers for training and inference for weights or activations.
[`BaseKerasTrainableQuantizer`](base_keras_quantizer.py) constitutes a base class for trainable quantizers of specific of specific tasks - currently, [`BaseKerasQATTrainableQuantizer`](../../../qat/keras/quantizer/base_keras_qat_quantizer.py) for Quantization-Aware Training.

## Examples and Fully implementation quantizers
For fully reference, check our QAT quantizers here:
[QAT Quantizers](../../../qat/keras/README.md)
