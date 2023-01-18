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
# ==============================================================================f


from model_compression_toolkit.core.common.constants import FOUND_TF
from model_compression_toolkit.core.common.logger import Logger
from model_compression_toolkit.quantizers_infrastructure.common.node_quantization_dispatcher import \
    NodeQuantizationDispatcher

if FOUND_TF:
    import tensorflow as tf
    from tensorflow.python.util import tf_inspect
    from tensorflow_model_optimization.python.core.keras import utils

    keras = tf.keras

    DISPATCHER = "dispatcher"
    LAYER = "layer"
    STEPS = "optimizer_step"
    TRAINING = "training"


    def _make_quantizer_fn(quantizer, x, training):
        """Use currying to return True/False specialized fns to the cond."""

        def quantizer_fn():
            return quantizer(x, training)

        return quantizer_fn


    def _weight_name(name: str) -> str:
        """Extracts the weight name from the full TensorFlow variable name.

        For example, returns 'kernel' for 'dense_2/kernel:0'.

        Args:
          name: TensorFlow variable name.

        Returns:
          Extracted weight name.
        """
        return name.split(':')[0].split('/')[-1]


    class KerasQuantizationWrapper(tf.keras.layers.Wrapper):
        def __init__(self,
                     layer,
                     dispatcher: NodeQuantizationDispatcher,
                     **kwargs):
            """
            Keras Quantization Wrapper takes a keras layer and dispatcher and infer a quantized layer.

            Args:
                layer: A keras layer.
                dispatcher: A node quantization dispatcher.
            """
            super(KerasQuantizationWrapper, self).__init__(layer, **kwargs)
            self._track_trackable(layer, name='layer')
            self.dispatcher = dispatcher

        def get_config(self):
            """

            Returns: Configuration of KerasQuantizationWrapper.

            """
            base_config = super(KerasQuantizationWrapper, self).get_config()
            config = {DISPATCHER: keras.utils.serialize_keras_object(self.dispatcher)}
            return dict(list(base_config.items()) + list(config.items()))

        @classmethod
        def from_config(cls, config):
            """

            Args:
                config(dict): dictionary  of  KerasNodeQuantizationDispatcher Configuration

            Returns: A KerasNodeQuantizationDispatcher

            """
            config = config.copy()

            dispatcher = keras.utils.deserialize_keras_object(
                config.pop(DISPATCHER),
                module_objects=globals(),
                custom_objects=None)

            layer = tf.keras.layers.deserialize(config.pop(LAYER))

            return cls(layer=layer, dispatcher=dispatcher, **config)

        def build(self, input_shape):
            """
            KerasQuantization Wrapper build function.
            Args:
                input_shape: the layer input shape

            Returns: None

            """
            super(KerasQuantizationWrapper, self).build(input_shape)

            self.optimizer_step = self.add_weight(
                STEPS,
                initializer=tf.keras.initializers.Constant(-1),
                dtype=tf.dtypes.int32,
                trainable=False)

            self._weight_vars = []
            for name, quantizer in self.dispatcher.weight_quantizers.items():
                weight = getattr(self.layer, name)
                quantizer.initialize_quantization(weight.shape,
                                                  _weight_name(weight.name), self)

                self._weight_vars.append((name, weight, quantizer))
                self._trainable_weights.append(weight)

            self._activation_vars = []
            for i, quantizer in enumerate(self.dispatcher.activation_quantizers):
                quantizer.initialize_quantization(None,
                                                  self.layer.name + f'/out{i}', self)
                self._activation_vars.append(quantizer)

        def set_quantize_weights(self, quantized_weights: dict):
            """
            This function update layer weights after quantization.

            Args:
                quantized_weights: a dict of weight to update

            Returns: None

            """
            for weight_attr in self.dispatcher.weight_quantizers.keys():
                weight = quantized_weights.get(weight_attr)
                current_weight = getattr(self.layer, weight_attr)
                if current_weight.shape != weight.shape:
                    Logger.error(
                        f"Existing layer weight shape {current_weight.shape} is incompatible with provided weight "
                        f"shape {weight.shape}")  # pragma: no cover

                setattr(self.layer, weight_attr, weight)

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

            # Quantize all weights, and replace them in the underlying layer.
            quantized_weights = {}
            for name, unquantized_weight, quantizer in self._weight_vars:
                quantized_weight = utils.smart_cond(
                    training,
                    _make_quantizer_fn(quantizer, unquantized_weight, True),
                    _make_quantizer_fn(quantizer, unquantized_weight, False))
                quantized_weights.update({name: quantized_weight})

            self.set_quantize_weights(quantized_weights)

            args_spec = tf_inspect.getfullargspec(self.layer.call).args
            if TRAINING in args_spec:
                outputs = self.layer.call(inputs, training=training, **kwargs)
            else:
                outputs = self.layer.call(inputs, **kwargs)

            # Quantize all activations if quantizers exist.
            if self.dispatcher.is_activation_quantization:
                num_outputs = len(outputs) if isinstance(outputs, (list, tuple)) else 1
                if self.dispatcher.num_act_quantizers != num_outputs:
                    Logger.error('Quantization wrapper output quantization error: '
                                 f'number of outputs and quantizers mismatch ({num_outputs}!='
                                 f'{self.dispatcher.num_act_quantizers}')
                if num_outputs == 1:
                    outputs = [outputs]

                _outputs = []
                for _output, act_quant in zip(outputs, self.dispatcher.activation_quantizers):
                    _outputs.append(utils.smart_cond(
                        training,
                        _make_quantizer_fn(act_quant, _output, True),
                        _make_quantizer_fn(act_quant, _output, False)))

                outputs = _outputs[0] if num_outputs == 1 else _outputs

            return outputs

else:
    class KerasQuantizationWrapper(object):
        def __init__(self, layer, dispatcher: NodeQuantizationDispatcher):
            """
            Keras Quantization Wrapper takes a keras layer and dispatcher and infer a quantized layer.

            Args:
                layer: A keras layer.
                dispatcher: A node quantization dispatcher.
            """
            Logger.critical('Installing tensorflow and tensorflow_model_optimization is mandatory '
                            'when using KerasQuantizationWrapper. '
                            'Could not find Tensorflow package.')
