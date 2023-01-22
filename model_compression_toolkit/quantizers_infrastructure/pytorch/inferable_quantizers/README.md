## Introduction

The `BasePyTorchInferableQuantizer` class is a base class for PyTorch quantizers that are used for inference only. It is a subclass of `BaseInferableQuantizer` and is designed to be used as a base class for creating custom quantization methods for PyTorch models.

## Installation

To use the `BasePyTorchInferableQuantizer` class, you will need to have PyTorch installed. If PyTorch is not installed, an exception will be raised.

Once you have PyTorch installed, you can use the `BasePyTorchInferableQuantizer` class by importing it from the package where it is located.

## Usage

The `BasePyTorchInferableQuantizer` class takes one argument during initialization:

- `quantization_target`: An enum which selects the quantizer tensor type: activation or weights.

Once you have instantiated the class, you can use the `__call__` method to quantize the given inputs using the quantizer parameters. The method takes one argument:

- `inputs`: input tensor to quantize

The method returns the quantized tensor.

```python
from torch_quantizer import BasePyTorchInferableQuantizer

quantizer = BasePyTorchInferableQuantizer(quantization_target='weights')
inputs = torch.randn((3,3))
outputs = quantizer(inputs)
print(outputs)
```

You must also implement the abstract method __call__ in your subclass which inherits BasePyTorchInferableQuantizer

```python
class MyQuantizer(BasePyTorchInferableQuantizer):
    def __init__(self,
                 quantization_target: QuantizationTarget):
        super(MyQuantizer, self).__init__(quantization_target=quantization_target)

    def __call__(self, inputs: torch.Tensor):
        quantized = inputs.round()
        return quantized

```

##Note

Keep in mind that BasePyTorchInferableQuantizer is an abstract class, it should not be instantiated directly. You should create a new class that inherits from it and implements the required methods.

If you have any questions or issues using the BasePyTorchInferableQuantizer class, please open an issue on the GitHub repository or reach out to the maintainers for assistance.