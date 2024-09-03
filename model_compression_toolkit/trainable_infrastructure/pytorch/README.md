## Introduction

[`BasePytorchTrainableQuantizer`](base_pytorch_quantizer.py) is an interface that enables easy quantizers development and training. 
Using this base class makes it simple to implement new quantizers for training and inference for weights or activations.
[`BasePytorchTrainableQuantizer`](base_pytorch_quantizer.py) constitutes a base class for trainable quantizers of specific tasks, such as QAT.

### The mark_quantizer Decorator

The [`@mark_quantizer`](https://github.com/sony/mct_quantizers/blob/main/mct_quantizers/common/base_inferable_quantizer.py) decorator is used to supply each quantizer with static properties which define its task compatibility. Each quantizer class should be decorated with this decorator. It defines the following properties:
 - [`QuantizationTarget`](https://github.com/sony/mct_quantizers/blob/main/mct_quantizers/common/base_inferable_quantizer.py): An Enum that indicates whether the quantizer is designated for weights or activations quantization.
 - [`QuantizationMethod`](https://github.com/sony/mct_quantizers/blob/main/mct_quantizers/common/quant_info.py): A list of quantization methods (Uniform, Symmetric, etc.).
 - `identifier`: A unique identifier for the quantizer class. This is a helper property that allows the creation of advanced quantizers for specific tasks.

Note that the `@mark_quantizer` decorator, and the `QuantizationTarget` and `QuantizationMethod` enums are provided by the external [MCT Quantizers](https://github.com/sony/mct_quantizers/) package.

## Examples and Fully implementation quantizers
Examples of Trainable Activation quantizers can be found here [Activation Quantizers](./activation_quantizers) and Trainable Weight quantizers here
[QAT Weight Quantizers](../../qat/pytorch/quantizer)
