## Introduction

PyTorch inferable quantizers are used for inference only. The inferable quantizer should contain all quantization
information needed for quantizing a PyTorch tensor. The quantization of the tensor can be done by calling the quantizer
while passing the unquantized tensor.

## Implemented PyTorch Inferable Quantizers

Several PyTorch inferable quantizers were implemented for activation quantization:

[ActivationPOTInferableQuantizer](activation_inferable_quantizers/activation_pot_inferable_quantizer.py)

[ActivationSymmetricInferableQuantizer](activation_inferable_quantizers/activation_symmetric_inferable_quantizer.py)

[ActivationUniformInferableQuantizer](activation_inferable_quantizers/activation_uniform_inferable_quantizer.py)

Each of them should be used according to the quantization method of the quantizer (power-of-two, symmetric and uniform
quantization respectively).

Similarly, several PyTorch inferable quantizers were implemented for weights quantization:

[WeightsPOTInferableQuantizer](weights_inferable_quantizers/weights_pot_inferable_quantizer.py)

[WeightsSymmetricInferableQuantizer](weights_inferable_quantizers/weights_symmetric_inferable_quantizer.py)

[WeightsUniformInferableQuantizer](weights_inferable_quantizers/weights_uniform_inferable_quantizer.py)

Each of them should be used according to the quantization method of the quantizer (power-of-two, symmetric and uniform
quantization respectively).

## Usage Example

```python
# Import PyTorch, Numpy and quantizers_infrastructure
import numpy as np
import torch

from model_compression_toolkit import quantizers_infrastructure as qi

# Create a weights symmetric quantizer for quantizing a kernel. The quantizer
# has the following properties:
# * It uses 8 bits for quantization.
# * It quantizes the tensor per-channel.
# Since it is a symmetric quantizer it needs to have the thresholds.
# Thus, the quantizer also:
# * Uses three thresholds (since it has 3 output channels and the quantization is per-channel): 1, 2 and 4.
# Notice that for weights we always use signed quantization.
quantizer = qi.pytorch_inferable_quantizers.WeightsSymmetricInferableQuantizer(num_bits=8,
                                                                               per_channel=True,
                                                                               threshold=np.asarray([2, 4, 1]),
                                                                               channel_axis=3)

# Get working device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize a random input to quantize between -50 to 50. Move the tensor to the working device
input_tensor = torch.rand(1, 3, 3, 3).to(device) * 100 - 50

# Quantize tensor
quantized_tensor = quantizer(input_tensor)
print(quantized_tensor)

# The maximal threshold is 4 using a signed quantization, so we expect all values to be in this range
assert torch.max(quantized_tensor) < 4, f'Quantized values should not contain values greater than maximal threshold'
assert torch.min(quantized_tensor) >= -4, f'Quantized values should not contain values lower than minimal threshold'

```

If you have any questions or issues using the PyTorch inferable quantizers, please open an issue on the GitHub.