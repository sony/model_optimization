from typing import Dict, Callable

import torch
import numpy as np

from model_compression_toolkit.core.common.constants import SIGNED, CLUSTER_CENTERS, THRESHOLD, MULTIPLIER_N_BITS, EPS
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor


def activation_lut_kmean_quantizer(activation_n_bits: int,
                                   quantization_params: Dict[str, np.ndarray]) -> Callable:
    """
    Builds a LUT quantizer for layer's activation using the provided params (threshold and clusters).
    It initiates a fake custom LUT layer that provides the quantizer function.

    Args:
        activation_n_bits: Number of bits to use for quantization (not used in this function).
        quantization_params: Dictionary of specific parameters for this quantization function.

    Returns:
        A fake LUT quantization node.
    """

    lut_fake_quant = PytorchLUTFakeQuant(quantization_params=quantization_params)
    return lambda x: lut_fake_quant(x)


class PytorchLUTFakeQuant(torch.nn.Module):
    """
    A custom PyTorch layer for quantizing activation tensor with non-uniform quantization (using lookup table clustering).
    """

    def __init__(self,
                 quantization_params: Dict[str, np.ndarray]):
        """
        Construct a Pytorch module that quantizes an activation tensor.

        Args:
            quantization_params: Dictionary of specific parameters for this quantization function.
        """

        super(PytorchLUTFakeQuant, self).__init__()

        self.quantization_params = quantization_params
        self.activation_is_signed = self.quantization_params.get(SIGNED)
        self.cluster_centers = to_torch_tensor(self.quantization_params.get(CLUSTER_CENTERS))
        self.threshold = self.quantization_params.get(THRESHOLD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize the output of an activation,

        Args:
            x: input tensor (which is the activation output of the layer that we want to quantize)

        Returns:
            Quantized torch Tensor.
        """
        if self.activation_is_signed is None or self.cluster_centers is None or self.threshold is None:
            return None

        _quant_output = self.lut_kmeans_quantizer(x)
        return _quant_output

    def lut_kmeans_quantizer(self, tensor_data: torch.Tensor) -> torch.Tensor:
        """
        Quantize a tensor using a non-uniform quantization based on the pre-defined kmeans clusters.
        1. Scales tensor_data with the threshold into 8-bit quantization range.
        2. Assigns cluster centers to each value.
        3. Scales back by multiplying the result by threshold and dividing with the quantization range max value.
        The result is the quantized tensor.

        Args:
            tensor_data: Input activation tensor.

        Returns: Quantized tensor.
        """

        tensor = self.int_quantization_with_threshold(tensor_data, MULTIPLIER_N_BITS)
        tensor = tensor.unsqueeze(-1)

        expanded_cluster_centers = self.cluster_centers.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
        cluster_assignments = torch.argmin(torch.abs(tensor - expanded_cluster_centers), dim=-1)
        centers = self.cluster_centers.flatten()[cluster_assignments]

        quant_tensor = (centers / (2 ** (MULTIPLIER_N_BITS - int(self.activation_is_signed)))) * self.threshold

        return quant_tensor

    def int_quantization_with_threshold(self,
                                        data: torch.Tensor,
                                        n_bits: int,
                                        eps: float = EPS) -> torch.Tensor:
        """
        Divides data by threshold and quantize it to integers in the quantization range (depends on signed value).

        Args:
            data: tensor data.
            n_bits: number of bits that determines the quantization range.
            eps: Small value for numerical stability in division.

        Returns:
            Uniform Quantized tensor.

        """

        if self.activation_is_signed:
            clip_max = 2 ** (n_bits - 1) - 1
            clip_min = -2 ** (n_bits - 1)
        else:
            clip_max = 2 ** n_bits - 1
            clip_min = 0

        return torch.clip((data / (self.threshold + eps)) * (2 ** (n_bits - int(self.activation_is_signed))),
                          min=clip_min, max=clip_max)
