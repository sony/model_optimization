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

from model_compression_toolkit.quantizers_infrastructure.common.constants import FOUND_TF
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import QuantizationTarget

if FOUND_TF:
    import tensorflow as tf
    from model_compression_toolkit.quantizers_infrastructure.keras.quantizer_utils import lut_kmeans_quantizer
    from model_compression_toolkit.quantizers_infrastructure.keras.inferable_quantizers \
        .base_lut_pot_inferable_quantizer import BaseLutPOTInferableQuantizer


    class WeightsLutPOTInferableQuantizer(BaseLutPOTInferableQuantizer):
        """
        Class for quantizing weights using a lut pot quantizer
        """

        def __init__(self,
                     num_bits: int,
                     cluster_centers: np.ndarray,
                     threshold: np.ndarray,
                     signed: bool,
                     ):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                cluster_centers: the cluster centers to assign the weights
                threshold: threshold for quantizing the weights
                signed: whether or not to use signed quantization
            """
            super(WeightsLutPOTInferableQuantizer, self).__init__(num_bits,
                                                                  cluster_centers,
                                                                  threshold,
                                                                  signed,
                                                                  QuantizationTarget.Weights)

        def __call__(self, inputs: tf.Tensor):
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize

            Returns:
                quantized tensor.
            """
            _quant_output = lut_kmeans_quantizer(inputs, cluster_centers=self.cluster_centers, signed=self.signed,
                                                 threshold=self.threshold)
            return _quant_output

        def get_config(self):
            """
            Return a dictionary with the configuration of the quantizer.

            Returns:
                Dictionary with the following keys: 'num_bits', 'cluster_centers', 'threshold', 'signed'
            """
            return {'num_bits': self.num_bits,
                    'cluster_centers': self.cluster_centers,
                    'threshold': self.threshold,
                    'signed': self.signed}


else:
    class WeightsLutPOTInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing tensorflow and tensorflow_model_optimization is mandatory '
                            'when using WeightsLutPOTInferableQuantizer. '
                            'Could not find Tensorflow package.')  # pragma: no cover
