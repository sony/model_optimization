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

from model_compression_toolkit.core.common.constants import FOUND_TF
from model_compression_toolkit.quantizers_infrastructure import BaseInferableQuantizer

if FOUND_TF:
    import tensorflow as tf

    class BaseKerasInferableQuantizer(BaseInferableQuantizer):
        def __init__(self):
            """
            This class is a base quantizer for Keras quantizers for inference only.
            """
            super(BaseKerasInferableQuantizer, self).__init__()

        @abstractmethod
        def get_config(self):
            """
            Return a dictionary with the configuration of the quantizer.
            """
            raise NotImplemented(f'{self.__class__.__name__} did not implement get_config')

        @abstractmethod
        def __call__(self, inputs: tf.Tensor):
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize

            Returns:
                quantized tensor.
            """
            raise NotImplemented(f'{self.__class__.__name__} did not implement __call__')
else:
    class BaseKerasInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing tensorflow and tensorflow_model_optimization is mandatory '
                            'when using BaseKerasInferableQuantizer. '
                            'Could not find Tensorflow package.')


