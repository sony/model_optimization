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
import torch

from model_compression_toolkit.core.common.constants import FOUND_TORCH
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import QuantizationTarget

if FOUND_TORCH:
    from model_compression_toolkit.quantizers_infrastructure.pytorch.inferable_quantizers \
        .base_lut_sym_inferable_quantizer import \
        BaseLutSymInferableQuantizer


    class BaseLutPOTInferableQuantizer(BaseLutSymInferableQuantizer):

        def __init__(self,
                     num_bits: int,
                     cluster_centers: np.ndarray,
                     threshold: np.ndarray,
                     signed: bool,
                     quantization_target: QuantizationTarget):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                cluster_centers: the cluster centers to assign the values
                threshold: threshold for quantizing values
                signed: whether or not to use signed quantization
                quantization_target: An enum which selects the quantizer tensor type: activation or weights.
            """

            super(BaseLutPOTInferableQuantizer, self).__init__(num_bits=num_bits,
                                                               cluster_centers=cluster_centers,
                                                               threshold=threshold,
                                                               signed=signed,
                                                               quantization_target=quantization_target)

            is_threshold_pot = np.all([(int(torch.log2(x)) == torch.log2(x)).detach().cpu().numpy()
                                       for x in self.threshold.flatten()])
            assert is_threshold_pot, f'Expected threshold to be power of 2 but is ' \
                                     f'{int(self.threshold.detach().cpu().numpy())}'


else:
    class BaseLutPOTInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using BaseLutPOTInferableQuantizer. '
                            'Could not find torch package.')  # pragma: no cover
