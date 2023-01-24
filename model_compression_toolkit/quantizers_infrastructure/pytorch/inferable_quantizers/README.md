## Introduction

PyTorch inferable quantizers are used for inference only. The inferable quantizer should contain all quantization information needed for quantizing a PyTorch tensor.
The quantization of the tensor can be done by calling the quantizer while passing the unquantized tensor.

## Implemented PyTorch Inferable Quantizers

Several PyTorch inferable quantizers were implemented for activation quantization:
```markdown
ActivationPOTInferableQuantizer
ActivationSymmetricInferableQuantizer
ActivationUniformInferableQuantizer
```
Each of them should be used according to the quantization method of the quantizer (power-of-two, symmetric and uniform quantization respectively).

Similarly, several PyTorch inferable quantizers were implemented for weights quantization:
```markdown
WeightsPOTInferableQuantizer
WeightsSymmetricInferableQuantizer
WeightsUniformInferableQuantizer
```
Each of them should be used according to the quantization method of the quantizer (power-of-two, symmetric and uniform quantization respectively).

## Usage Example

```python
# Import PyTorch and quantizers_infrastructure
import torch

from model_compression_toolkit import quantizers_infrastructure as qi

# Create a weights symmetric quantizer for quantizing a kernel using 8 bits,
# quantization per-channel, 3 threshold (in this example, the layer's kernel
# outputs 3 channels, thus we're using 3 thresholds - threshold per channel)
# and the quantization is signed.
quantizer = qi.pytorch_inferable_quantizers.WeightsSymmetricInferableQuantizer(num_bits=8,
                                                                               threshold=torch.Tensor([2, 4, 1]),
                                                                               per_channel=True,
                                                                               signed=True)

# Get working device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize a random input to quantize between -50 to 50. Move the tensor to the working device
input_tensor = torch.rand(1, 3, 3, 3).to(device) * 100 - 50

# Quantize tensor
quantized_tensor = quantizer(input_tensor)
print(quantized_tensor)
assert torch.max(quantized_tensor) < 4, f'Quantized values should not contain values greater than maximal threshold'
assert torch.min(quantized_tensor) >= -4, f'Quantized values should not contain values lower than minimal threshold'

```

If you have any questions or issues using the PyTorch inferable quantizers, please open an issue on the GitHub.