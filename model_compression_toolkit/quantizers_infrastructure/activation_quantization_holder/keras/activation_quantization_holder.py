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
import tensorflow as tf

from model_compression_toolkit.constants import FOUND_TF
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.quantizers_infrastructure import BaseInferableQuantizer
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.common.constants import ACTIVATION_QUANTIZER

if FOUND_TF:
    keras = tf.keras

    class ActivationQuantizationHolder(keras.layers.Layer):
        """
        Keras layer to hold an activation quantizer and quantize during inference.
        """
        def __init__(self,
                     activation_quantizer: BaseInferableQuantizer,
                     **kwargs):
            """

            Args:
                activation_quantizer: Quantizer to use during inference.
                **kwargs: Key-word arguments for the base layer
            """

            super(ActivationQuantizationHolder, self).__init__(**kwargs)
            self.activation_quantizer = activation_quantizer

        def get_config(self):
            """
            Returns: Configuration of ActivationQuantizationHolder.

            """
            base_config = super(ActivationQuantizationHolder, self).get_config()
            config = {
                ACTIVATION_QUANTIZER: keras.utils.serialize_keras_object(self.activation_quantizer)}

            return dict(list(base_config.items()) + list(config.items()))

        @classmethod
        def from_config(cls, config):
            """

            Args:
                config(dict): dictionary  of  ActivationQuantizationHolder Configuration

            Returns: A ActivationQuantizationHolder object

            """
            config = config.copy()
            activation_quantizer = keras.utils.deserialize_keras_object(config.pop(ACTIVATION_QUANTIZER),
                                                                        module_objects=globals(),
                                                                        custom_objects=None)

            return cls(activation_quantizer=activation_quantizer, **config)

        def call(self, inputs):
            """
            ActivationQuantizationHolder call function

            Args:
                inputs: Input tensors to quantize use the activation quantizer the object holds

            Returns: Output of the activation quantizer

            """
            return self.activation_quantizer(inputs)

else:
    class ActivationQuantizationHolder:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            Logger.error('Installing tensorflow and tensorflow_model_optimization is mandatory '
                         'when using ActivationQuantizationHolder. '
                         'Could not find Tensorflow package.')