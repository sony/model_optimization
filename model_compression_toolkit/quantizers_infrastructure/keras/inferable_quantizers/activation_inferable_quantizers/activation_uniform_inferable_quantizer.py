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

from model_compression_toolkit.core.common.logger import Logger
from model_compression_toolkit.core.common.constants import FOUND_TF
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.quantizers_infrastructure import QuantizationTarget
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import mark_quantizer

if FOUND_TF:
    import tensorflow as tf
    from model_compression_toolkit.quantizers_infrastructure.keras.inferable_quantizers.base_uniform_inferable_quantizer import BaseUniformInferableQuantizer

    @mark_quantizer(quantization_target=QuantizationTarget.Activation,
                    quantization_method=[QuantizationMethod.UNIFORM],
                    quantizer_type=None)
    class ActivationUniformInferableQuantizer(BaseUniformInferableQuantizer):
        """
        Class for quantizing activations using an uniform quantizer
        """

        def __init__(self,
                     num_bits: int,
                     min_range: List[float],
                     max_range: List[float],
                     ):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                min_range: min range for quantizing activations
                max_range: max range for quantizing activations
            """
            # Call the superclass constructor with the given parameters, along with the target of Activation
            # quantization
            super(ActivationUniformInferableQuantizer, self).__init__(num_bits,
                                                                      min_range,
                                                                      max_range)

            assert len(self.min_range) == 1, f'In per-tensor quantization min_range should be of length 1 but is {len(self.min_range)}'
            assert len(self.max_range) == 1, f'In per-tensor quantization max_range should be of length 1 but is {len(self.max_range)}'

            self.min_range = self.min_range[0]
            self.max_range = self.max_range[0]

        def __call__(self, inputs:tf.Tensor) -> tf.Tensor:
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize

            Returns:
                quantized tensor.
            """
            assert inputs.dtype==tf.float32, f'Input tensor was expected to be a float tensor but is of type {inputs.dtype}'

            return tf.quantization.fake_quant_with_min_max_vars(inputs,
                                                                min=self.min_range,
                                                                max=self.max_range,
                                                                num_bits=self.num_bits)

        def get_config(self):
            """
            Return a dictionary with the configuration of the quantizer.

            Returns:
                Dictionary with the following keys: 'num_bits', 'min_range', 'max_range'
            """
            return {'num_bits': self.num_bits,
                    'min_range': self.min_range,
                    'max_range': self.max_range}

else:
    class ActivationUniformInferableQuantizer:
        def __init__(self, *args, **kwargs):
            Logger.error('Installing tensorflow and tensorflow_model_optimization is mandatory '
                         'when using ActivationUniformInferableQuantizer. '
                         'Could not find Tensorflow package.')  # pragma: no cover
