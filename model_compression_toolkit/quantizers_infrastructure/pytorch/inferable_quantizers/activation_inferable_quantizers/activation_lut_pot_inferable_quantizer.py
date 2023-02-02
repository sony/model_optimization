# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np

from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import QuantizationTarget
from model_compression_toolkit.core.common.constants import FOUND_TORCH
from model_compression_toolkit.quantizers_infrastructure.pytorch.quantizer_utils import lut_kmeans_quantizer

if FOUND_TORCH:
    import torch
    from model_compression_toolkit.quantizers_infrastructure.pytorch.inferable_quantizers \
        .base_lut_pot_inferable_quantizer import BaseLutPOTInferableQuantizer


    class ActivationLutPOTInferableQuantizer(BaseLutPOTInferableQuantizer):
        """
        Class for quantizing activations using Lut power-of-two quantizer
        """

        def __init__(self,
                     num_bits: int,
                     cluster_centers: np.ndarray,
                     threshold: np.ndarray,
                     signed: bool
                     ):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                cluster_centers: the cluster centers to assign the values
                threshold: threshold for quantizing activations
                signed: whether or not to use signed quantization
            """
            # target of Activation quantization
            super(ActivationLutPOTInferableQuantizer, self).__init__(num_bits,
                                                                     cluster_centers,
                                                                     threshold,
                                                                     signed,
                                                                     QuantizationTarget.Activation)

        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize

            Returns:
                quantized tensor.
            """
            inputs.requires_grad = False
            _quant_output = lut_kmeans_quantizer(inputs, cluster_centers=self.cluster_centers, signed=self.signed,
                                                 threshold=self.threshold)
            return _quant_output


else:
    class ActivationLutPOTInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using ActivationLutPOTInferableQuantizer. '
                            'Could not find torch package.')  # pragma: no cover
