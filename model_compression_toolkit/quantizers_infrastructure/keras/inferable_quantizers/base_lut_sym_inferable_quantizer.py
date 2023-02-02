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

from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import QuantizationTarget
from model_compression_toolkit.quantizers_infrastructure.keras.inferable_quantizers.base_keras_inferable_quantizer \
    import BaseKerasInferableQuantizer


class BaseLutSymInferableQuantizer(BaseKerasInferableQuantizer):

    def __init__(self,
                 num_bits: int,
                 cluster_centers: np.ndarray,
                 threshold: np.ndarray,
                 signed: bool,
                 quantization_target: QuantizationTarget
                 ):
        """
        Initialize the quantizer with the specified parameters.

        Args:
            num_bits: number of bits to use for quantization
            cluster_centers: the cluster centers to assign the values
            threshold: threshold for quantizing activations
            signed: whether or not to use signed quantization
            quantization_target: An enum which selects the quantizer tensor type: activation or weights.
        """
        super(BaseLutSymInferableQuantizer, self).__init__(quantization_target=quantization_target)
        self.num_bits = num_bits
        self.cluster_centers = cluster_centers
        self.threshold = threshold
        self.signed = signed

    @abstractmethod
    def get_config(self):
        """
        Return a dictionary with the configuration of the quantizer.
        """
        raise NotImplemented(f'{self.__class__.__name__} did not implement get_config')  # pragma: no cover
