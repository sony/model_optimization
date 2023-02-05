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
from abc import abstractmethod

import numpy as np

from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import mark_quantizer
from model_compression_toolkit.quantizers_infrastructure.keras.inferable_quantizers.base_uniform_inferable_quantizer \
    import \
    BaseUniformInferableQuantizer


@mark_quantizer(quantization_target=None,
                quantization_method=[QuantizationMethod.SYMMETRIC],
                quantizer_type=None)
class BaseSymmetricInferableQuantizer(BaseUniformInferableQuantizer):

    def __init__(self,
                 num_bits: int,
                 threshold: np.ndarray,
                 signed: bool):
        """
        Initialize the quantizer with the specified parameters.

        Args:
            num_bits: number of bits to use for quantization
            threshold: threshold for quantizing weights
            signed: whether or not to use signed quantization
        """

        if not isinstance(threshold, np.ndarray):
            threshold = np.asarray(threshold)

        self.signed = signed
        self.threshold = threshold

        delta = threshold / (2 ** (num_bits - int(signed)))
        min_range = -threshold if signed else 0
        max_range = threshold - delta

        super(BaseSymmetricInferableQuantizer, self).__init__(min_range=min_range,
                                                              max_range=max_range,
                                                              num_bits=num_bits)

    @abstractmethod
    def get_config(self):
        """
        Return a dictionary with the configuration of the quantizer.
        """
        raise NotImplemented(f'{self.__class__.__name__} did not implement get_config')  # pragma: no cover

