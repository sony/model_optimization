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
import keras.models
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from keras.engine.base_layer import Layer
from keras.engine.input_layer import InputLayer
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_configs import \
    NoOpQuantizeConfig

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.exporter.fully_quantized.keras.builder.quantize_config_to_node import \
    SUPPORTED_QUANTIZATION_CONFIG
from model_compression_toolkit.exporter.fully_quantized.keras.quantize_configs.weights_quantize_config import \
    WeightsQuantizeConfig
from model_compression_toolkit.exporter.target_platform_export.keras.exporters.base_keras_exporter import BaseKerasExporter


class FakelyQuantKerasExporter(BaseKerasExporter):
    """
    Exporter for fakely-quant Keras models.
    """

    def __init__(self, model:keras.models.Model):
        super().__init__(model)

    def export(self) -> keras.models.Model:
        """
        Convert fully-quantized Keras model to a fakely-quant export-ready model.

        Returns:
            Fake-quant Keras model ready to export.
        """

        def _unwrap_quantize_wrapper(layer:Layer):
            """
            Convert layer wrapped with QuantizeWrapper to the layer it wraps, so it's ready to export.

            Args:
                layer: Layer to unwrap.

            Returns:
                Layer after unwrapping.

            """

            if isinstance(layer, tfmot.quantization.keras.QuantizeWrapper):
                # For now, we assume only weights are quantized for models to export. Activation support is in progress.
                if isinstance(layer.layer, InputLayer):
                    if not isinstance(layer.quantize_config, NoOpQuantizeConfig):
                        Logger.error(
                            f'Currently, FakelyQuantKerasExporter supports weights quantization only. Please consider '
                            f'disabling activation quantization, until supported ')
                    return Layer() #Identity

                if not isinstance(layer.quantize_config, tuple(SUPPORTED_QUANTIZATION_CONFIG)):
                    Logger.error(
                        f'Only supported quantization configs are {tuple(SUPPORTED_QUANTIZATION_CONFIG)}')

                # If weights are quantized, use the quantized weight for the new built layer.
                if isinstance(layer.quantize_config, WeightsQuantizeConfig):
                    new_layer = layer.layer.__class__.from_config(layer.layer.get_config())
                    with tf.name_scope(new_layer.name):
                        new_layer.build(layer.input_shape)

                    # Build a list of the layer's new weights.
                    weights_list = []
                    for w in new_layer.weights:
                        val = None
                        for qw in layer.weights:
                            if w.name in qw.name:
                                if w.name.split('/')[-1].split(':')[0] in layer.quantize_config.weight_attrs:
                                    val = layer.quantize_config.get_weights_and_quantizers(layer.layer)[0][1].weight
                                else:
                                    val = qw
                        if val is None:
                            Logger.error(f'Could not match weight name: {w.name}')
                        weights_list.append(val)

                    new_layer.set_weights(weights_list)
                    new_layer.trainable = False
                    return new_layer

                elif isinstance(layer.quantize_config, NoOpQuantizeConfig):
                    return layer.layer

                else:
                    Logger.error(
                        f'Currently, FakelyQuantKerasExporter supports weights quantization only. Please consider '
                        f'disabling activation quantization, until supported ')
            else:
                return layer

        # clone each layer in the model and apply _unwrap_quantize_wrappera to layers wrapped with a QuantizeWrapper.
        self.exported_model = tf.keras.models.clone_model(self.model,
                                                          input_tensors=None,
                                                          clone_function=_unwrap_quantize_wrapper)

        return self.exported_model

    def save_model(self, save_model_path:str):
        """
        Save exported model to a given path.
        Args:
            save_model_path: Path to save the model.

        Returns:
            None.
        """
        if self.exported_model is None:
            Logger.critical(f'Exporter can not save model as it is not exported')
        keras.models.save_model(self.exported_model, save_model_path)
