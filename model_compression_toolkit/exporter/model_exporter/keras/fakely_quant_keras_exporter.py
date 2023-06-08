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

import keras
import keras.models
import keras.models
import tensorflow as tf
from keras.engine.base_layer import Layer

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.exporter.model_exporter.keras.base_keras_exporter import \
    BaseKerasExporter
from mct_quantizers import KerasQuantizationWrapper

layers = keras.layers

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

        if self.exported_model is None:
            Logger.critical(f'Exporter can not save model as it is not exported')  # pragma: no cover

        Logger.info(f'Exporting FQ Keras model to: {self.save_model_path}')

        keras.models.save_model(self.exported_model, self.save_model_path)

        return FakelyQuantKerasExporter.get_custom_objects()

