## Introduction

[`BaseKerasTrainableQuantizer`](base_keras_quantizer.py) is an interface that enables easy quantizers development and training. 
Using this base class makes it simple to implement new quantizers for training and inference for weights or activations.
[`BaseKerasTrainableQuantizer`](base_keras_quantizer.py) constitutes a base class for trainable quantizers of specific of specific tasks - currently, [`BaseKerasQATTrainableQuantizer`](../../../qat/keras/quantizer/base_keras_qat_quantizer.py) for Quantization-Aware Training.

### The mark_quantizer Decorator
The [`@mark_quantizer`](../../inferable_infrastructure/common/base_inferable_quantizer.py) decorator is used to supply each quantizer with static properties which define its task compitability. Each quantizer class should be decorated with this decorator. It defines the following properties:
 - [`QuantizationTarget`](../../inferable_infrastructure/common/base_inferable_quantizer.py): And Enum that indicates whether the quantizer is designated for weights or activations quantization.
 - [`QuantizationMethod`](../../../core/common/target_platform/op_quantization_config.py): List of quantization methods (Uniform, Symmetric, etc.).
 - `quantizer_type`: An Enum defines the type of the quantization technique (varies between different quantization tasks).

## Examples and Fully implementation quantizers
For fully reference, check our QAT quantizers here:
[QAT Quantizers](../../../qat/keras/README.md)
