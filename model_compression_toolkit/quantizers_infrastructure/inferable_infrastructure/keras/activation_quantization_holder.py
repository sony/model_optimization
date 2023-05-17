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

from model_compression_toolkit.constants import FOUND_TF
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.quantizers_infrastructure import BaseInferableQuantizer, BaseKerasTrainableQuantizer
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.common.constants import \
    ACTIVATION_HOLDER_QUANTIZER, TRAINING, STEPS

if FOUND_TF:
    import tensorflow as tf
    from keras.utils import tf_inspect
    from tensorflow_model_optimization.python.core.keras import utils

    def _make_quantizer_fn(quantizer, x, training):
        """Use currying to return True/False specialized fns to the cond."""

        def quantizer_fn():
            return quantizer(x, training)

        return quantizer_fn

    keras = tf.keras

    class ActivationQuantizationHolder(keras.layers.Layer):
        """
        Keras layer to hold an activation quantizer and quantize during inference.
        """
        def __init__(self,
                     activation_holder_quantizer: BaseInferableQuantizer,
                     **kwargs):
            """

            Args:
                activation_holder_quantizer: Quantizer to use during inference.
                **kwargs: Key-word arguments for the base layer
            """

            super(ActivationQuantizationHolder, self).__init__(**kwargs)
            self.activation_holder_quantizer = activation_holder_quantizer

        def get_config(self):
            """
            Returns: Configuration of ActivationQuantizationHolder.

            """
            base_config = super(ActivationQuantizationHolder, self).get_config()
            config = {
                ACTIVATION_HOLDER_QUANTIZER: keras.utils.serialize_keras_object(self.activation_holder_quantizer)}

            return dict(list(base_config.items()) + list(config.items()))

        @classmethod
        def from_config(cls, config):
            """

            Args:
                config(dict): dictionary  of ActivationQuantizationHolder Configuration

            Returns: A ActivationQuantizationHolder object

            """
            config = config.copy()
            activation_holder_quantizer = keras.utils.deserialize_keras_object(config.pop(ACTIVATION_HOLDER_QUANTIZER),
                                                                               module_objects=globals(),
                                                                               custom_objects=None)

            return cls(activation_holder_quantizer=activation_holder_quantizer,
                       **config)

        def build(self, input_shape):
            """
            ActivationQuantizationHolder build function.
            Args:
                input_shape: the layer input shape

            Returns: None

            """
            super(ActivationQuantizationHolder, self).build(input_shape)

            self.optimizer_step = self.add_weight(
                STEPS,
                initializer=tf.keras.initializers.Constant(-1),
                dtype=tf.dtypes.int32,
                trainable=False)

            self.activation_holder_quantizer.initialize_quantization(None,
                                                                     self.name + '/out_',
                                                                     self)

        def call(self,
                 inputs: tf.Tensor,
                 training=None) -> tf.Tensor:
            """
            Quantizes the input tensor using the activation quantizer the ActivationQuantizationHolder holds.

            Args:
                inputs: Input tensors to quantize use the activation quantizer the object holds
                training: a boolean stating if layer is in training mode.

            Returns: Output of the activation quantizer (quantized input tensor).

            """
            if training is None:
                training = tf.keras.backend.learning_phase()

            activation_quantizer_args_spec = tf_inspect.getfullargspec(self.activation_holder_quantizer.__call__).args
            if TRAINING in activation_quantizer_args_spec:
                return utils.smart_cond(
                    training,
                    _make_quantizer_fn(self.activation_holder_quantizer, inputs, True),
                    _make_quantizer_fn(self.activation_holder_quantizer, inputs, False))

            return self.activation_holder_quantizer(inputs)

        def convert_to_inferable_quantizers(self):
            """
            Convert layer's quantizer to inferable quantizer.

            Returns:
                None
            """
            if isinstance(self.activation_holder_quantizer, BaseKerasTrainableQuantizer):
                self.activation_holder_quantizer = self.activation_holder_quantizer.convert2inferable()




else:
    class ActivationQuantizationHolder:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            Logger.error('Installing tensorflow and tensorflow_model_optimization is mandatory '
                         'when using ActivationQuantizationHolder. '
                         'Could not find Tensorflow package.')