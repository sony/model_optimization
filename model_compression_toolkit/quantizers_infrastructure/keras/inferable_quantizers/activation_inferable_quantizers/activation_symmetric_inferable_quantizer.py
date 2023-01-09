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

from model_compression_toolkit.core.common.constants import FOUND_TF
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import QuantizationTarget

if FOUND_TF:
    import tensorflow as tf

    from model_compression_toolkit.quantizers_infrastructure.keras.inferable_quantizers \
        .base_symmetric_inferable_quantizer import \
        BaseSymmetricInferableQuantizer


    class ActivationSymmetricInferableQuantizer(BaseSymmetricInferableQuantizer):
        """
        Class for quantizing activations using a symmetric quantizer
        """

        def __init__(self,
                     num_bits: int,
                     threshold: np.ndarray,
                     signed: bool):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                threshold: threshold for quantizing activations
                signed: whether or not to use signed quantization
            """
            # Call the superclass constructor with the given parameters, along with the target of Activation
            # quantization
            super(ActivationSymmetricInferableQuantizer, self).__init__(num_bits=num_bits,
                                                                        threshold=threshold,
                                                                        signed=signed,
                                                                        quantization_target=QuantizationTarget.Activation)

        def get_config(self):
            """
            Return a dictionary with the configuration of the quantizer.

            Returns:
                Dictionary with the following keys: 'num_bits', 'signed', 'threshold'
            """
            return {'num_bits': self.num_bits,
                    'signed': self.signed,
                    'threshold': self.threshold}

        def __call__(self, inputs, training=False):
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize
                training: whether or not the quantizer is being used in training mode (unused here)

            Returns:
                quantized tensor.
            """
            return tf.quantization.fake_quant_with_min_max_vars(inputs,
                                                                min=self.min_range,
                                                                max=self.max_range,
                                                                num_bits=self.num_bits)

else:
    class ActivationSymmetricInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing tensorflow and tensorflow_model_optimization is mandatory '
                            'when using ActivationSymmetricInferableQuantizer. '
                            'Could not find Tensorflow package.')
