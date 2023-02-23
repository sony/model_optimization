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
from typing import List

import numpy as np

from model_compression_toolkit.core.common.constants import FOUND_TF
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.common.base_inferable_quantizer import mark_quantizer, \
    QuantizationTarget
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.common.constants import MULTIPLIER_N_BITS, EPS

if FOUND_TF:
    from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.keras.quantizers.weights_inferable_quantizers.\
        weights_lut_symmetric_inferable_quantizer import WeightsLUTSymmetricInferableQuantizer


    @mark_quantizer(quantization_target=QuantizationTarget.Weights,
                    quantization_method=[QuantizationMethod.LUT_POT_QUANTIZER],
                    quantizer_type=None)
    class WeightsLUTPOTInferableQuantizer(WeightsLUTSymmetricInferableQuantizer):
        """
        Class for quantizing weights using a lut power-of-two quantizer
        """

        def __init__(self,
                     num_bits: int,
                     cluster_centers: np.ndarray,
                     threshold: List[float],
                     per_channel: bool,
                     channel_axis: int = None,
                     input_rank: int = None,
                     multiplier_n_bits: int = MULTIPLIER_N_BITS,
                     eps: float = EPS):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                cluster_centers: the cluster centers to assign the weights
                threshold: threshold for quantizing weights
                per_channel: whether to use per-channel quantization
                channel_axis: axis along which to apply per-channel quantization
                input_rank: number of dimensions of input tensor the quantizer quantizes
                multiplier_n_bits: Number of bits that determines the quantization range
                eps: Small value for numerical stability in division
            """

            super(WeightsLUTPOTInferableQuantizer, self).__init__(num_bits=num_bits,
                                                                  cluster_centers=cluster_centers,
                                                                  threshold=threshold,
                                                                  per_channel=per_channel,
                                                                  channel_axis=channel_axis,
                                                                  input_rank=input_rank,
                                                                  multiplier_n_bits=multiplier_n_bits,
                                                                  eps=eps)

            is_threshold_pot = np.all([int(np.log2(x)) == np.log2(x) for x in self.threshold.flatten()])
            assert is_threshold_pot, f'Expected threshold to be power of 2 but is {self.threshold}'


else:
    class WeightsLUTPOTInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing tensorflow and tensorflow_model_optimization is mandatory '
                            'when using WeightsLUTPOTInferableQuantizer. '
                            'Could not find Tensorflow package.')
