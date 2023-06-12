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
import copy
from typing import Callable

import keras.models
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, Reshape
from keras.models import clone_model

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.exporter.model_exporter.keras.fakely_quant_keras_exporter import FakelyQuantKerasExporter
from mct_quantizers import constants as keras_inferable_constants, KerasQuantizationWrapper

BIAS_INITIALIZER = 'bias_initializer'
BIAS_REGULARIZER = 'bias_regularizer'
BIAS_CONSTRAINT = 'bias_constraint'
ACTIVITY_REGULARIZER = 'activity_regularizer'
KERNEL_INITIALIZER = 'kernel_initializer'
KERNEL_REGULARIZER = 'kernel_regularizer'
KERNEL_CONSTRAINT = 'kernel_constraint'
KERNEL_SIZE = 'kernel_size'
PADDING = 'padding'
STRIDES = 'strides'
LAYER_NAME = 'name'
TRAINABLE = 'trainable'
ACTIVATION = 'activation'
USE_BIAS = 'use_bias'
FILTERS = 'filters'
UNITS = 'units'
PAD_VALID = 'valid'
KERNEL = 'kernel'

CONV_KERNEL_CHANNEL_AXIS = 3
CONV_INPUT_CHANNELS_DIM = 4


class INT8TFLiteExporter(FakelyQuantKerasExporter):
    """
    Exporter for INT8 TFLite models.
    The exporter expects to receive an exportable model (where each layer's full quantization parameters
    can be retrieved), and convert it into a quantized model where weights and activations are represented
    as integer data type.
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

        self.exported_model = None

    def _get_pointwise_layer_to_replace_dense(self, wrapped_layer: KerasQuantizationWrapper) -> keras.layers.Layer:
        # First we create a pointwise configuration based on the Dense layer's configuration
        dense_cfg = wrapped_layer.layer.get_config()

        # List of pw attributes that should be taken from the dense layer as they are.
        pw_attr_list = [LAYER_NAME, ACTIVATION, USE_BIAS, BIAS_CONSTRAINT,
                        BIAS_INITIALIZER, BIAS_REGULARIZER, TRAINABLE, ACTIVITY_REGULARIZER,
                        KERNEL_INITIALIZER, KERNEL_REGULARIZER, KERNEL_CONSTRAINT]

        pw_cfg = {attr: dense_cfg[attr] for attr in pw_attr_list}

        # Use more attributes that are not taken as they are
        pw_cfg.update({KERNEL_SIZE: (1, 1),
                       STRIDES: (1, 1),
                       PADDING: PAD_VALID,
                       FILTERS: dense_cfg[UNITS]})

        # Create the point-wise layer
        pw_layer = Conv2D(**pw_cfg)
        pw_layer.build(wrapped_layer.input_shape)

        # Create and set the point-wise weights to assign
        dense_kernel = wrapped_layer.layer.kernel
        pw_weights = []
        pw_kernel = np.reshape(wrapped_layer.get_weights()[0],
                               (1, 1, dense_kernel.get_shape()[0], dense_cfg[UNITS]))

        pw_weights.append(pw_kernel)
        if wrapped_layer.layer.use_bias:
            pw_bias = wrapped_layer.get_weights()[2]
            pw_weights.append(pw_bias)

        pw_layer.set_weights(pw_weights)

        # Now that we have the point-wise to replace the dense layer,
        # we need to wrap it using KerasQuantizationWrapper with a new
        # relevant quantizers.
        # Create new kernel quantizer
        pw_kernel_quantizer_cfg = wrapped_layer.weights_quantizers[KERNEL].get_config()

        # In Conv2D channel axis is 3 and not 1 as in Dense
        pw_kernel_quantizer_cfg[keras_inferable_constants.CHANNEL_AXIS] = CONV_KERNEL_CHANNEL_AXIS

        # Unquantized weight to conv layer has 4 dimensions (unlike dense which varies)
        pw_kernel_quantizer_cfg[keras_inferable_constants.INPUT_RANK] = CONV_INPUT_CHANNELS_DIM

        assert isinstance(pw_kernel_quantizer_cfg[keras_inferable_constants.THRESHOLD],
                          list), f'Expected to find threshold which is a list, but found: {type(pw_kernel_quantizer_cfg[keras_inferable_constants.THRESHOLD])}'
        pw_kernel_quantizer_cfg[keras_inferable_constants.THRESHOLD] = list(
            pw_kernel_quantizer_cfg[keras_inferable_constants.THRESHOLD])

        # Now that we have the point-wise quantizer we can instantiate it
        quantizer_class = type(wrapped_layer.weights_quantizers[KERNEL])
        pw_quantizer = quantizer_class(**pw_kernel_quantizer_cfg)
        pw_weights_quantizers = copy.deepcopy(wrapped_layer.weights_quantizers)
        pw_weights_quantizers[KERNEL] = pw_quantizer

        # Wrap pw with the new quantizers (the activation is not affected thus we take the Dense quantizers)
        wrapped_pw = KerasQuantizationWrapper(pw_layer,
                                              pw_weights_quantizers)

        # Compute the shape that the input to the new layer should be reshaped into
        # Example: Dense kernel with the following shape (3, 20) expects to have input with the
        # next dimensions (BATCH_SIZE, x0, x1, ..., xn, 20).
        # Conv layer expects 4-rank input. Thus, the input is reshaped to (BATCH_SIZE, 1, x0*x1*...*xn, 20)
        dim = wrapped_layer.input_shape[1:-1]
        target_shape = (1, int(np.prod(dim))) + (dense_kernel.get_shape()[0],)

        return Sequential([
            Reshape(target_shape=target_shape),
            wrapped_pw,
            Reshape(wrapped_layer.output_shape[1:])
        ])

    def export(self) -> None:
        """
        Export a fully quantized model to its int8 tflite model.
        """

        def _substitute_model(layer_to_substitue: keras.layers.Layer) -> keras.layers.Layer:
            assert self.is_layer_exportable_fn(
                layer_to_substitue), f'Layer {layer_to_substitue.get_config()} did not pass validation'

            # In order to support dense quantization using per-channel quantization (which is
            # unsupported in TFLITE int models) we substitute each dense layer to its equivalent
            # point-wise convolution.
            if isinstance(layer_to_substitue, KerasQuantizationWrapper):
                if isinstance(layer_to_substitue.layer, Dense):
                    return self._get_pointwise_layer_to_replace_dense(layer_to_substitue)

            return layer_to_substitue

        # Transform the model to a new model that can be converted to int8 models.
        # For example: replace dense layers with point-wise layers (to support per-channel quantization)
        self.transformed_model = clone_model(self.model,
                                             clone_function=_substitute_model)

        # Convert model to int8 representation
        converter = tf.lite.TFLiteConverter.from_keras_model(self.transformed_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.exported_model = converter.convert()

        Logger.info(f'Exporting INT8 tflite model to: {self.save_model_path}')
        with open(self.save_model_path, 'wb') as f:
            f.write(self.exported_model)
