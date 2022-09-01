# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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

import torch

from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import uniform_quantize_tensor
from model_compression_toolkit.core.common.target_platform import QuantizationMethod


class SymmetricWeightsQuantizer(torch.nn.Module):
    def __init__(self,
                 num_bits: int,
                 min_range,
                 max_range,
                 quantization_method: QuantizationMethod,
                 per_channel: bool,
                 output_channels_axis: int
                 ):
        """
        Symmetric WeightsQuantizer a Pytorch model that constitutes as a wrapper for a Pytorch layer, built from a
        given graph node.
        Args:
            n: Node to build its Pytorch layer.

        """

        super(SymmetricWeightsQuantizer, self).__init__()

        self.num_bits = num_bits
        self.min_range = min_range
        self.max_range = max_range
        self.quantization_method = quantization_method
        self.signed = True
        self.per_channel = per_channel
        self.output_channels_axis = output_channels_axis

    def __call__(self, float_weight, *args, **kwargs):
        return uniform_quantize_tensor(tensor_data=float_weight,
                                       range_min=self.min_range,
                                       range_max=self.max_range,
                                       n_bits=self.num_bits)
