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
import warnings

from model_compression_toolkit.core.common.constants import FOUND_TORCH
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.common.base_inferable_quantizer \
    import mark_quantizer

if FOUND_TORCH:
    from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.pytorch.quantizers \
        .base_pytorch_inferable_quantizer import BasePyTorchInferableQuantizer


    @mark_quantizer(quantization_target=None,
                    quantization_method=[QuantizationMethod.LUT_SYM_QUANTIZER],
                    quantizer_type=None)
    class BaseLUTSymmetricInferableQuantizer(BasePyTorchInferableQuantizer):

        def __init__(self,
                     num_bits: int,
                     cluster_centers: np.ndarray,
                     threshold: np.ndarray,
                     signed: bool,
                     multiplier_n_bits: int,
                     eps: float):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                cluster_centers: the cluster centers to assign the values
                threshold: threshold for quantizing values
                signed: whether or not to use signed quantization
                multiplier_n_bits: Number of bits that determines the quantization range
                eps: Small value for numerical stability in division
            """

            super(BaseLUTSymmetricInferableQuantizer, self).__init__()

            assert isinstance(threshold,
                              np.ndarray), f'Threshold is expected to be numpy array, but is of type {type(threshold)}'
            assert threshold.ndim == 1, f'Threshold is expected to be flatten, but of shape {threshold.shape}'

            assert len(np.unique(cluster_centers)) <= 2 ** num_bits, \
                f'Expected num of cluster centers to be less or equal than {2 ** num_bits} ' \
                f'but got {len(cluster_centers)}'

            assert not np.any(cluster_centers - cluster_centers.astype(int)), f'Expected cluster centers to be integers'

            if signed:
                assert np.all((-1 * (2 ** (multiplier_n_bits - int(signed))) <= cluster_centers) &
                              (cluster_centers <= (2 ** (multiplier_n_bits - int(signed)) - 1))), \
                    f'Expected cluster centers in the quantization range'
            else:
                assert np.all(cluster_centers <= (2 ** multiplier_n_bits)), f'Expected cluster centers in the ' \
                                                                            f'quantization range'

            # If unsigned activation quantization, all cluster_centers must be positive
            if not signed:
                assert np.all(cluster_centers >= 0), f'Expected unsigned cluster centers in unsigned activation ' \
                                                          f'quantization'

            # num_bits must be less than multiplier_n_bits
            assert num_bits <= multiplier_n_bits, f'Look-Up-Table bit configuration has {num_bits} bits. It must be ' \
                                                  f'less then {multiplier_n_bits}'
            if num_bits == multiplier_n_bits:
                warnings.warn("Num of bits equal to multiplier n bits, Please be aware LUT quantizier may be "
                              "inefficient in that case, consider using SymmetricInferableQuantizer instead")

            self.signed = signed
            self.threshold = threshold
            self.cluster_centers = cluster_centers
            self.num_bits = num_bits
            self.multiplier_n_bits = multiplier_n_bits
            self.eps = eps

else:
    class BaseLUTSymmetricInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory when using BaseLUTSymmetricInferableQuantizer. Could not '
                            'find torch package.')
