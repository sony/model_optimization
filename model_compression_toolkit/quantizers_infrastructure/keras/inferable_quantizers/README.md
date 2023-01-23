## Introduction

The `BaseKerasInferableQuantizer` class is a base class for Keras quantizers that are used for inference only. It is a subclass of `BaseInferableQuantizer` and is designed to be used as a base class for creating custom quantization methods for Keras models.

## Installation

To use the `BaseKerasInferableQuantizer` class, you will need to have `tensorflow`  and `tensorflow_model_optimization` installed. If one of them is not installed, an exception will be raised.

Once you have `tensorflow`  and `tensorflow_model_optimization` installed, you can use the `BaseKerasInferableQuantizer` class by importing it and implementing a quantizer.

## Usage

The `BaseKerasInferableQuantizer` class takes one argument during initialization:

- `quantization_target`: An enum which selects the quantizer tensor type: activation or weights.

Once you have instantiated your Keras inferable quantizer, you can use the `__call__` method to quantize the given inputs using the quantizer parameters. The method takes one argument:

- `inputs`: input tensor to quantize

The method returns the quantized tensor.

You must implement the abstract method `__call__` in your subclass which inherits BaseKerasInferableQuantizer

Another abstract method that should be implemented is `get_config` which returns a dictionary of the arguments that should be passed to the quantizer `__init__` method.

For example:

```python
import tensorflow as tf
from model_compression_toolkit import quantizers_infrastructure as qi


class MyQuantizer(qi.BaseKerasInferableQuantizer):
    def __init__(self,
                 quantization_target: qi.QuantizationTarget):
        super(MyQuantizer, self).__init__(quantization_target=quantization_target)

    def get_config(self):
        return {'quantization_target': self.quantization_target}

    def __call__(self, inputs: tf.Tensor):
        # Your quantization logic here
        return tf.round(inputs)


quantization_target = qi.QuantizationTarget.Activation
quantizer = MyQuantizer(quantization_target=quantization_target)
input_tensor = tf.random.normal(shape=(1, 10))
quantized_tensor = quantizer(input_tensor)
print(quantized_tensor)
```

## Note

Keep in mind that BaseKerasInferableQuantizer is an abstract class, it should not be instantiated directly. You should create a new class that inherits from it and implements the required methods.

If you have any questions or issues using the BaseKerasInferableQuantizer class, please open an issue on the GitHub repository or reach out to the maintainers for assistance.