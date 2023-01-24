## Introduction

Keras inferable quantizers are used for inference only. The inferable quantizer should contain all quantization information needed for quantizing a TensorFlow tensor.
The quantization of the tensor can be done by calling the quantizer while passing the unquantized tensor.

## Implemented Keras Inferable Quantizers

Several Keras inferable quantizers were implemented for activation quantization:
```markdown
[ActivationPOTInferableQuantizer](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/quantizers_infrastructure/keras/inferable_quantizers/activation_inferable_quantizers/activation_pot_inferable_quantizer.py)
ActivationSymmetricInferableQuantizer
ActivationUniformInferableQuantizer
```
Each of them should be used according to the quantization method of the quantizer (power-of-two, symmetric and uniform quantization respectively).

Similarly, several Keras inferable quantizers were implemented for weights quantization:
```markdown
WeightsPOTInferableQuantizer
WeightsSymmetricInferableQuantizer
WeightsUniformInferableQuantizer
```
Each of them should be used according to the quantization method of the quantizer (power-of-two, symmetric and uniform quantization respectively).

## Usage Example

```python
# Import TensorFlow and quantizers_infrastructure
import tensorflow as tf

from model_compression_toolkit import quantizers_infrastructure as qi

# Create a weights symmetric quantizer for quantizing a kernel using 8 bits,
# quantization per-channel (the channel index is passed as channel_axis, when -1 is the last axis),
# 3 threshold (in this example, the layer's kernel outputs 3 channels, thus we're
# using 3 thresholds - threshold per channel) and the quantization is signed.
quantizer = qi.keras_inferable_quantizers.WeightsSymmetricInferableQuantizer(num_bits=8,
                                                                             threshold=[2, 4, 1],
                                                                             per_channel=True,
                                                                             channel_axis=-1,
                                                                             signed=True)

# Initialize a random input qo quantize
input_tensor = tf.random.uniform(shape=(1, 3, 3, 3), minval=-100, maxval=100)
# Quantize tensor
quantized_tensor = quantizer(input_tensor)
print(quantized_tensor)
assert tf.reduce_max(quantized_tensor) < 4, f'Quantized values should not contain values greater than maximal threshold'
assert tf.reduce_min(quantized_tensor) >= -4, f'Quantized values should not contain values lower than minimal threshold'


```

If you have any questions or issues using the Keras inferable quantizers, please open an issue on the GitHub.
