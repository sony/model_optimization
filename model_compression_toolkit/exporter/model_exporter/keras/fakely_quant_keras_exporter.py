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
from typing import Dict, Callable

import keras.models
import keras.models
import tensorflow as tf
from keras.engine.base_layer import Layer

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.exporter.model_exporter.keras.base_keras_exporter import \
    BaseKerasExporter
from model_compression_toolkit.quantizers_infrastructure import KerasQuantizationWrapper, \
    KerasNodeQuantizationDispatcher



class FakelyQuantKerasExporter(BaseKerasExporter):
    """
    Exporter for fakely-quant Keras models.
    The exporter expects to receive an exportable model (where each layer's full quantization parameters
    can be retrieved), and convert it into a fakely-quant model (namely, weights that are in fake-quant
    format) and fake-quant layers for the activations.
    """

    def __init__(self,
                 model: keras.models.Model,
                 is_layer_exportable_fn: Callable,
                 save_model_path: str):
        """

        Args:
            model: Model to export.
            is_layer_exportable_fn: Callable to check whether a layer can be exported or not.
            save_model_path: Path to save the exported model.
        """

        super().__init__(model,
                         is_layer_exportable_fn,
                         save_model_path)

    def export(self) -> Dict[str, type]:
        """
        Convert an exportable (fully-quantized) Keras model to a fakely-quant model
        (namely, weights that are in fake-quant format) and fake-quant layers for the activations.
        """

        def _unwrap_quantize_wrapper(layer: Layer):
            """
            Convert layer wrapped with ExtendedQuantizeWrapper to the layer it wraps,
             so it's ready to export. Notice that layers with activation quantization remain
             wrapped using ExtendedQuantizeWrapper to contain its activation quantizer.

            Args:
                layer: Layer to unwrap.

            Returns:
                Layer after unwrapping.

            """
            assert self.is_layer_exportable_fn(layer), f'Layer {layer.name} is not exportable.'

            # If weights are quantized, use the quantized weight for the new built layer.
            if layer.dispatcher.is_weights_quantization:
                new_layer = layer.layer.__class__.from_config(layer.layer.get_config())
                with tf.name_scope(new_layer.name):
                    new_layer.build(layer.input_shape)

                # Build a list of the layer's new weights.
                weights_list = []
                # Go over weights, check if they should be quantized, and quantize if this is the case:
                for w in new_layer.weights:
                    val = None
                    for qw in layer.weights:
                        if w.name in qw.name:
                            # Use quantized weight if layer attribute should be quantized.
                            # For example: check if 'kernel_0' is an attribute
                            # that should be quantized. First, extract 'kernel' from variable name, check if the
                            # quantize config contains this as an attribute for quantization. If so -
                            # Take the quantized weight from the quantize_config and set it to the new layer.
                            attribute_name = w.name.split('/')[-1].split(':')[0]
                            if attribute_name in layer.dispatcher.weight_quantizers.keys():
                                quantizer = layer.dispatcher.weight_quantizers.get(attribute_name)
                                val = quantizer(qw, False)
                            else:
                                val = qw
                    if val is None:
                        Logger.error(f'Could not match weight name: {w.name}')
                    weights_list.append(val)

                new_layer.set_weights(weights_list)
                new_layer.trainable = False

                # If activations are also quantized, wrap the layer back using ActivationQuantizeConfig
                # from original wrapper (weights wrapping is no longer needed).
                if layer.dispatcher.is_activation_quantization:
                    activation_dispatcher = KerasNodeQuantizationDispatcher(weight_quantizers={},
                                                                            activation_quantizers=layer.dispatcher.activation_quantizers)
                    new_layer = KerasQuantizationWrapper(layer=new_layer,
                                                         dispatcher=activation_dispatcher)

                return new_layer

            # If this is a layer with activation quantization only, just return it
            # as activation quantization in the fake-quant case uses the wrapper for quantization.
            return layer


        # clone each layer in the model and apply _unwrap_quantize_wrapper to layers wrapped with a QuantizeWrapper.
        self.exported_model = tf.keras.models.clone_model(self.model,
                                                          input_tensors=None,
                                                          clone_function=_unwrap_quantize_wrapper)

        if self.exported_model is None:
            Logger.critical(f'Exporter can not save model as it is not exported')

        Logger.info(f'Exporting FQ Keras model to: {self.save_model_path}')

        keras.models.save_model(self.exported_model, self.save_model_path)

        return FakelyQuantKerasExporter.get_custom_objects()

