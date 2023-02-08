## Introduction

Keras inferable quantizers are used for inference only. The inferable quantizer should contain all quantization information needed for quantizing a TensorFlow tensor.
The quantization of the tensor can be done by calling the quantizer while passing the unquantized tensor.

## Implemented Keras Inferable Quantizers

Several Keras inferable quantizers were implemented for activation quantization:

[ActivationPOTInferableQuantizer](activation_inferable_quantizers/activation_pot_inferable_quantizer.py)

[ActivationSymmetricInferableQuantizer](activation_inferable_quantizers/activation_symmetric_inferable_quantizer.py)

[ActivationUniformInferableQuantizer](activation_inferable_quantizers/activation_uniform_inferable_quantizer.py)

Each of them should be used according to the quantization method of the quantizer (power-of-two, symmetric and uniform quantization respectively).

Similarly, several Keras inferable quantizers were implemented for weights quantization:

[WeightsPOTInferableQuantizer](weights_inferable_quantizers/weights_pot_inferable_quantizer.py)

[WeightsSymmetricInferableQuantizer](weights_inferable_quantizers/weights_symmetric_inferable_quantizer.py)

[WeightsUniformInferableQuantizer](weights_inferable_quantizers/weights_uniform_inferable_quantizer.py)

Each of them should be used according to the quantization method of the quantizer (power-of-two, symmetric and uniform quantization respectively).

## Usage Example

Here, we can see a demonstration of a symmetric weights quantizer usage:

```python
# Import TensorFlow and quantizers_infrastructure
import tensorflow as tf
from model_compression_toolkit import quantizers_infrastructure as qi

# Creates a WeightsSymmetricInferableQuantizer instance for quantizing a tensor with four dimensions.
# The quantizer uses 8 bits for quantization and quantizes the tensor per channel.
# The quantizer uses three thresholds (1, 2, and 3) for quantizing each of the three output channels.
# The quantization axis is the last dimension (-1).
quantizer = qi.keras_inferable_quantizers.WeightsSymmetricInferableQuantizer(num_bits=8,
                                                                             threshold=[2.0, 3.0, 1.0],
                                                                             per_channel=True,
                                                                             channel_axis=-1,
                                                                             input_rank=4)

# Initialize a random input to quantize. Note that the input must have float data type.
input_tensor = tf.random.uniform(shape=(1, 3, 3, 3), minval=-100, maxval=100, dtype=tf.float32)
# Quantize tensor
quantized_tensor = quantizer(input_tensor)
print(quantized_tensor)

# The maximal threshold is 3 using a signed quantization, so we expect all values to be in this range
assert tf.reduce_max(quantized_tensor) < 3, f'Quantized values should not contain values greater than maximal threshold'
assert tf.reduce_min(quantized_tensor) >= -3, f'Quantized values should not contain values lower than minimal threshold'

```

Now, let's see a demonstration of a symmetric activation quantizer usage:

```python
# Import TensorFlow and quantizers_infrastructure
import tensorflow as tf
from model_compression_toolkit import quantizers_infrastructure as qi

# Creates an ActivationSymmetricInferableQuantizer quantizer.
# The quantizer uses 8 bits for quantization and quantizes the tensor per-tensor 
# (per-channel quantization is not supported in activation quantizers). 
# The quantization is unsigned, meaning the range of values is between 0 and the
# threshold, which is 5.0.
quantizer = qi.keras_inferable_quantizers.ActivationSymmetricInferableQuantizer(num_bits=8,
                                                                                threshold=[5.0],
                                                                                signed=False)

# Initialize a random input to quantize. Note that the input must have float data type.
input_tensor = tf.random.uniform(shape=(1, 3, 3, 3), minval=-100, maxval=100, dtype=tf.float32)
# Quantize tensor
quantized_tensor = quantizer(input_tensor)
print(quantized_tensor)

# The maximal threshold is 3 using a signed quantization, so we expect all values to be in this range
assert tf.reduce_max(quantized_tensor) < 5, f'Quantized values should not contain values greater than maximal threshold'
assert tf.reduce_min(quantized_tensor) >= 0, f'Quantized values should not contain values lower than minimal threshold'

```

If you have any questions or issues using the Keras inferable quantizers, please [open an issue](https://github.com/sony/model_optimization/issues/new/choose) in this GitHub repository.
