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
from tensorflow.python.util import tf_inspect

from model_compression_toolkit.quantizers_infrastructure import BaseInferableQuantizer
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.common.constants import \
    TRAINING, ACTIVATION_QUANTIZER

keras = tf.keras


class ActivationQuantizationHolder(keras.layers.Layer):
    def __init__(self,
                 activation_quantizer: BaseInferableQuantizer,
                 **kwargs):

        super(ActivationQuantizationHolder, self).__init__(**kwargs)
        self.activation_quantizer = activation_quantizer

    def get_config(self):
        """
        Returns: Configuration of KerasQuantizationWrapper.

        """
        base_config = super(ActivationQuantizationHolder, self).get_config()
        config = {
            ACTIVATION_QUANTIZER: keras.utils.serialize_keras_object(self.activation_quantizer)}

        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        """

        Args:
            config(dict): dictionary  of  KerasQuantizationWrapper Configuration

        Returns: A KerasQuantizationWrapper

        """
        config = config.copy()
        activation_quantizer = keras.utils.deserialize_keras_object(config.pop(ACTIVATION_QUANTIZER),
                                                                    module_objects=globals(),
                                                                    custom_objects=None)

        return cls(activation_quantizer=activation_quantizer, **config)

    def call(self, inputs, training=None, **kwargs):
        """
        KerasQuantizationWrapper call functions
        Args:
            inputs: Input tensors to specified layer
            training: a boolean stating if layer is in training mode.
            **kwargs:

        Returns: tensors that simulate a quantized layer.

        """
        if training is None:
            training = tf.keras.backend.learning_phase()

        args_spec = tf_inspect.getfullargspec(self.layer.call).args
        if TRAINING in args_spec:
            outputs = self.activation_quantizer.call(inputs, training=training, **kwargs)
        else:
            outputs = self.activation_quantizer.call(inputs, **kwargs)

        return outputs
