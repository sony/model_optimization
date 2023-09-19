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
import copy
import importlib
from typing import Dict, Callable

import keras.models
import tensorflow as tf
from packaging import version
if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.engine.base_layer import Layer
else:
    from keras.engine.base_layer import Layer

from mct_quantizers import KerasQuantizationWrapper

layers = keras.layers

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.exporter.model_exporter.keras.base_keras_exporter import BaseKerasExporter

from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras import backend


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

            # Assert each layer is exportable
            self.is_layer_exportable_fn(layer)

            # If weights are quantized, use the quantized weight for the new built layer.
            if isinstance(layer, KerasQuantizationWrapper):
                if layer.is_weights_quantization:
                    new_layer = layer.layer.__class__.from_config(layer.layer.get_config())

                    # Build a list of the layer's new weights.
                    weights_list = []

                    # Create a list of weights for the new created layer
                    if isinstance(layer.layer, layers.DepthwiseConv2D):
                        weights_list.append(layer.get_quantized_weights()['depthwise_kernel'])
                    elif isinstance(layer.layer, (layers.Conv2D, layers.Dense, layers.Conv2DTranspose)):
                        weights_list.append(layer.get_quantized_weights()['kernel'])
                    else:
                        Logger.error(f'KerasQuantizationWrapper should wrap only DepthwiseConv2D, Conv2D, Dense'
                                     f' and Conv2DTranspose layers but wrapped layer is {layer.layer}')

                    if layer.layer.bias is not None:
                        weights_list.append(layer.layer.bias)

                    # In order to add the weights of the layer, we need to build it. To build it
                    # we need to pass its input shape. Not every layer has input_shape since some
                    # layers may have multiple inputs with different input shapes (reused layers for
                    # example). For this reason, we take input shape at index 0 (any input shape
                    # should work since the weights are dependent only at some dimensions which have to
                    # be the same for all inputs).
                    with tf.name_scope(new_layer.name):
                        new_layer.build(layer.get_input_shape_at(0))

                    new_layer.set_weights(weights_list)
                    new_layer.trainable = False

                    return new_layer

            return layer

        # clone each layer in the model and apply _unwrap_quantize_wrapper to layers wrapped with a QuantizeWrapper.
        self.exported_model = tf.keras.models.clone_model(self.model,
                                                          input_tensors=None,
                                                          clone_function=_unwrap_quantize_wrapper)

        # Transform the model's configuration to convert KerasActivationQuantizationHolder to fake-quant layers
        new_cfg = self.transform_model_cfg()

        # Create a new model with the transformed configuration
        new_model = tf.keras.Model().from_config(new_cfg)

        # Filter "optimization step" variables from the weights of the model (they are used
        # in KerasActivationQuantizationHolder only for training)
        filtered_weights = self.get_filtered_weights()
        new_model.set_weights(filtered_weights)
        self.exported_model = new_model

        if self.exported_model is None:
            Logger.critical(f'Exporter can not save model as it is not exported')  # pragma: no cover

        Logger.info(f'Exporting FQ Keras model to: {self.save_model_path}')

        keras.models.save_model(self.exported_model, self.save_model_path)

        return FakelyQuantKerasExporter.get_custom_objects()

    def transform_model_cfg(self) -> dict:
        """
       Transforms the model configuration, convert KerasActivationQuantizationHolder layers
       to fake-quant layers.
       """
        # Get the existing configuration of the exported model
        old_cfg = self.exported_model.get_config()

        new_layers_cfg = []
        for old_layer in old_cfg['layers']:
            # Transform KerasActivationQuantizationHolder to a fake-quant
            if old_layer['class_name'] == 'KerasActivationQuantizationHolder':
                # In order to create the fake-quant layer call_kwargs (such as min/max/num_bits) we ideally
                # want to extract it from the layer configuration. However, the layer configuration
                # does not contain this information, but different information such as threshold and signedness.
                # For this reason, we instantiate the quantizer that is holded (by the
                # KerasActivationQuantizationHolder) and then extract this information from it.
                # First, we get the quantizer class from mct_quantizers
                new_layer_cfg = copy.deepcopy(old_layer)
                module_mct_quantizers = importlib.import_module('mct_quantizers')
                keras_quantizers_module = getattr(module_mct_quantizers, 'keras_quantizers')
                quantizer_class = getattr(keras_quantizers_module, f'{old_layer["config"]["activation_holder_quantizer"]["class_name"]}')

                # Then, we instantiate the quantizer with the layer's configuration.
                quantizer = quantizer_class(**old_layer['config']['activation_holder_quantizer']['config'])

                assert len(quantizer.min_range) == 1, f'Activation quantizers support only per-tensor quantization, ' \
                                                      f' thus min should be of length 1 but is ' \
                                                      f'{len(quantizer.min_range)} in layer ' \
                                                      f'{old_layer["config"]["name"]}'

                assert len(quantizer.max_range) == 1, f'Activation quantizers support only per-tensor quantization, ' \
                                                      f' thus max should be of length 1 but is ' \
                                                      f'{len(quantizer.max_range)} in layer {old_layer["config"]["name"]}'

                fake_quant_layer_call_kwargs = {'min': quantizer.min_range[0],
                                                'max': quantizer.max_range[0],
                                                'num_bits': quantizer.num_bits,
                                                'narrow_range': False,
                                                'name': None}

                # Create fake-quant layer configuration
                new_layer_cfg['class_name'] = 'TFOpLambda'
                new_layer_cfg['config'] = {'name': old_layer['config']['name'],
                                           'trainable': False,
                                           'dtype': 'float32',
                                           'function': 'quantization.fake_quant_with_min_max_vars'}

                # Assume new_layer_cfg['inbound_nodes'] is a list in the following format: [[[X, X, X, X]]]
                # TFOpLambda should have 'inbound_nodes' in the format: [[X, X, X, X]]
                # In addition, the 4th element (new_layer_cfg['inbound_nodes'][0][3]) should contain the
                # call_kwargs dictionary of the layer (the rest first 3 elements contain details about the
                # connectivity of the layer's previous layers).
                new_layer_cfg['inbound_nodes'] = new_layer_cfg['inbound_nodes'][0]
                new_layer_cfg['inbound_nodes'][0][3] = fake_quant_layer_call_kwargs
                new_layers_cfg.append(new_layer_cfg)
            else:
                # Keep non-KerasActivationQuantizationHolder layers unchanged
                new_layers_cfg.append(old_layer)

        new_cfg = copy.deepcopy(old_cfg)
        new_cfg['layers'] = new_layers_cfg

        return new_cfg

    def get_filtered_weights(self):
        """
        Retrieve and filter the weights of the model excluding optimizer-related weights.
        """
        output_weights = []

        # Iterate through each weight in the model
        for single_w in self.exported_model.weights:

            # Exclude optimizer-related weights
            if 'optimizer_step' not in single_w.name:
                if isinstance(single_w, base_layer_utils.TrackableWeightHandler):
                    output_weights.extend(single_w.get_tensors())
                else:
                    output_weights.append(single_w)

        # Retrieve the values of the filtered weights
        filtered_weights = backend.batch_get_value(output_weights)

        return filtered_weights
