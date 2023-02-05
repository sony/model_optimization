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
from typing import Dict, List
from model_compression_toolkit import quantizers_infrastructure as qi
from model_compression_toolkit.core.common.constants import FOUND_TF
from model_compression_toolkit.core.common.logger import Logger
from model_compression_toolkit.quantizers_infrastructure import BaseInferableQuantizer
from model_compression_toolkit.quantizers_infrastructure.common.constants import WEIGHTS_QUANTIZERS, ACTIVATION_QUANTIZERS, LAYER, STEPS, TRAINING

if FOUND_TF:
    import tensorflow as tf
    from tensorflow.python.util import tf_inspect
    from tensorflow_model_optimization.python.core.keras import utils

    keras = tf.keras

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
                     weights_quantizers: Dict[str, BaseInferableQuantizer] = None,
                     activation_quantizers: List[BaseInferableQuantizer] = None,
                     **kwargs):
            """
            Keras Quantization Wrapper takes a keras layer and quantizers and infer a quantized layer.

            Args:
                layer: A keras layer.
                weights_quantizers: A dictionary between a weight's name to its quantizer.
                activation_quantizers: A list of activations quantization, one for each layer output.
            """
            super(KerasQuantizationWrapper, self).__init__(layer, **kwargs)
            self._track_trackable(layer, name='layer')
            self.weights_quantizers = weights_quantizers if weights_quantizers is not None else dict()
            self.activation_quantizers = activation_quantizers if activation_quantizers is not None else list()

        def add_weights_quantizer(self, param_name: str, quantizer: BaseInferableQuantizer):
            """
            This function adds a weights quantizer to existing wrapper

            Args:
                param_name: The name of the parameter to quantize
                quantizer: A quantizer.

            Returns: None

            """
            self.weights_quantizers.update({param_name: quantizer})

        @property
        def is_activation_quantization(self) -> bool:
            """
            This function check activation quantizer exists in wrapper.
            Returns: a boolean if activation quantizer exists

            """
            return self.num_activation_quantizers > 0

        @property
        def is_weights_quantization(self) -> bool:
            """
            This function check weights quantizer exists in wrapper.

            Returns: a boolean if weights quantizer exists

            """
            return self.num_weights_quantizers > 0

        @property
        def num_weights_quantizers(self) -> int:
            """
            Returns: number of weights quantizers
            """
            return len(self.weights_quantizers)

        @property
        def num_activation_quantizers(self) -> int:
            """
            Returns: number of activations quantizers
            """
            return len(self.activation_quantizers)

        def get_config(self):
            """
            Returns: Configuration of KerasQuantizationWrapper.

            """
            base_config = super(KerasQuantizationWrapper, self).get_config()
            config = {
                ACTIVATION_QUANTIZERS: [keras.utils.serialize_keras_object(act) for act in self.activation_quantizers],
                WEIGHTS_QUANTIZERS: {k: keras.utils.serialize_keras_object(v) for k, v in self.weights_quantizers.items()}}
            return dict(list(base_config.items()) + list(config.items()))

        def _set_weights_vars(self, is_training: bool = True):
            """
            This function sets weights quantizers vars to the layer

            Args:
                is_training: Flag to indicate whether training or not

            Returns: None
            """
            self._weights_vars = []
            for name, quantizer in self.weights_quantizers.items():
                weight = getattr(self.layer, name)
                quantizer.initialize_quantization(weight.shape, _weight_name(weight.name) if is_training else None,
                                                  self)
                self._weights_vars.append((name, weight, quantizer))
                self._trainable_weights.append(weight) # Must when inherit from tf.keras.layers.Wrapper in tf2.10 and below

        def _set_activations_vars(self):
            """
            This function sets activations quantizers vars to the layer

            Returns: None
            """
            self._activation_vars = []
            for i, quantizer in enumerate(self.activation_quantizers):
                quantizer.initialize_quantization(None, self.layer.name + f'/out{i}', self)
                self._activation_vars.append(quantizer)

        @classmethod
        def from_config(cls, config):
            """

            Args:
                config(dict): dictionary  of  KerasQuantizationWrapper Configuration

            Returns: A KerasQuantizationWrapper

            """
            config = config.copy()
            activation_quantizers = [keras.utils.deserialize_keras_object(act,
                                                                          module_objects=globals(),
                                                                          custom_objects=None) for act in
                                     config.pop(ACTIVATION_QUANTIZERS)]
            weights_quantizers = {k: keras.utils.deserialize_keras_object(v,
                                                                         module_objects=globals(),
                                                                         custom_objects=None) for k, v in
                                 config.pop(WEIGHTS_QUANTIZERS).items()}
            layer = tf.keras.layers.deserialize(config.pop(LAYER))
            return cls(layer=layer, weights_quantizers=weights_quantizers, activation_quantizers=activation_quantizers, **config)

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

            self._set_weights_vars()
            self._set_activations_vars()

        def set_quantize_weights(self, quantized_weights: dict):
            """
            This function update layer weights after quantization.

            Args:
                quantized_weights: a dict of weight to update

            Returns: None

            """
            for weight_attr in self.weights_quantizers.keys():
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
            for name, unquantized_weight, quantizer in self._weights_vars:

                weights_quantizer_args_spec = tf_inspect.getfullargspec(quantizer.__call__).args
                if TRAINING in weights_quantizer_args_spec:
                    quantized_weight = utils.smart_cond(
                        training,
                        _make_quantizer_fn(quantizer, unquantized_weight, True),
                        _make_quantizer_fn(quantizer, unquantized_weight, False))
                    quantized_weights.update({name: quantized_weight})
                else:
                    # Keras weights inferable quantizer
                    quantized_weight = quantizer(unquantized_weight)
                    quantized_weights.update({name: quantized_weight})

            self.set_quantize_weights(quantized_weights)

            args_spec = tf_inspect.getfullargspec(self.layer.call).args
            if TRAINING in args_spec:
                outputs = self.layer.call(inputs, training=training, **kwargs)
            else:
                outputs = self.layer.call(inputs, **kwargs)

            # Quantize all activations if quantizers exist.
            if self.is_activation_quantization:
                num_outputs = len(outputs) if isinstance(outputs, (list, tuple)) else 1
                if self.num_activation_quantizers != num_outputs:
                    Logger.error('Quantization wrapper output quantization error: '
                                 f'number of outputs and quantizers mismatch ({num_outputs}!='
                                 f'{self.num_activation_quantizers}')
                if num_outputs == 1:
                    outputs = [outputs]

                _outputs = []
                for _output, act_quant in zip(outputs, self.activation_quantizers):
                    activation_quantizer_args_spec = tf_inspect.getfullargspec(act_quant.__call__).args
                    if TRAINING in activation_quantizer_args_spec:
                        _outputs.append(utils.smart_cond(
                            training,
                            _make_quantizer_fn(act_quant, _output, True),
                            _make_quantizer_fn(act_quant, _output, False)))
                    else:
                        # Keras activation inferable quantizer.
                        _outputs.append(act_quant(_output))
                outputs = _outputs[0] if num_outputs == 1 else _outputs

            return outputs

        def convert_to_inferable_quantizers(self):
            """
            Convert layer's quantizers to inferable.

            Returns:
                None
            """
            # Activations quantizers
            inferable_activation_quantizers = []
            if self.is_activation_quantization:
                for quantizer in self.activation_quantizers:
                    if isinstance(quantizer, qi.BaseKerasTrainableQuantizer):
                        inferable_activation_quantizers.append(quantizer.convert2inferable())
                self.activation_quantizers = inferable_activation_quantizers
                self._set_activations_vars()

            # Weight quantizers
            inferable_weight_quantizers = {}
            if self.is_weights_quantization:
                for name, quantizer in self.weights_quantizers.items():
                    if isinstance(quantizer, qi.BaseKerasTrainableQuantizer):
                        inferable_weight_quantizers.update({name: quantizer.convert2inferable()})
                self.weights_quantizers = inferable_weight_quantizers
                self._set_weights_vars(False)

else:
    class KerasQuantizationWrapper(object):
        def __init__(self,
                     layer,
                     weights_quantizers: Dict[str, BaseInferableQuantizer] = None,
                     activation_quantizers: List[BaseInferableQuantizer] = None):
            """
            Keras Quantization Wrapper takes a keras layer and quantizers and infer a quantized layer.

            Args:
                layer: A keras layer.
                weights_quantizers: A dictionary between a weight's name to its quantizer.
                activation_quantizers: A list of activations quantization, one for each layer output.
            """
            Logger.critical('Installing tensorflow and tensorflow_model_optimization is mandatory '
                            'when using KerasQuantizationWrapper. '
                            'Could not find Tensorflow package.')  # pragma: no cover
