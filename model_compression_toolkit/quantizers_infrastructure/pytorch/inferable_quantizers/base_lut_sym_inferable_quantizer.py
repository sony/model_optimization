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

from model_compression_toolkit.core.common.constants import FOUND_TORCH
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import QuantizationTarget
from model_compression_toolkit.quantizers_infrastructure.pytorch.quantizer_utils import get_working_device, \
    to_torch_tensor

if FOUND_TORCH:
    from model_compression_toolkit.quantizers_infrastructure.pytorch.inferable_quantizers \
        .base_pytorch_inferable_quantizer import \
        BasePyTorchInferableQuantizer


    class BaseLutSymInferableQuantizer(BasePyTorchInferableQuantizer):

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

            super(BaseLutSymInferableQuantizer, self).__init__(quantization_target=quantization_target)

            assert isinstance(threshold,
                              np.ndarray), f'Threshold is expected to be numpy array, but is of type {type(threshold)}'
            assert threshold.ndim == 1, f'Threshold is expected to be flatten, but of shape {threshold.shape}'

            self.signed = signed
            self.cluster_centers = to_torch_tensor(cluster_centers).to(get_working_device())
            self.threshold = to_torch_tensor(threshold).to(get_working_device())
            self.num_bits = num_bits


else:
    class BaseLutSymInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using BaseLutSymInferableQuantizer. '
                            'Could not find torch package.')  # pragma: no cover
