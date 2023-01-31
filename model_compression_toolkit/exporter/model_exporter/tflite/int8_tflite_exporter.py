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
from typing import Callable

import keras.models
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, Reshape
from keras.models import clone_model

from model_compression_toolkit import quantizers_infrastructure as qi
from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.exporter.model_exporter.keras.fakely_quant_keras_exporter import FakelyQuantKerasExporter

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

class INT8TFLiteExporter(FakelyQuantKerasExporter):
    """
    Exporter for fakely-quant TFLite models.
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

        self.exported_model = None

    def export(self):
        """
        Convert an exportable (fully-quantized) Keras model to a fakely-quant TFLite model
        (namely, weights that are in fake-quant format) and fake-quant layers for the activations.

        """

        def _substitute_model(wrapped_layer):
            assert self.is_layer_exportable_fn(wrapped_layer), f'Layer {wrapped_layer.get_config()} did not pass validation'

            if isinstance(wrapped_layer.layer, Dense):
                # pw_kernel = self._convert_dense_kernel_to_pw_kernel(wrapped_layer)
                # List of pw attributes that should take from separable as they are.

                pw_attr_list = [LAYER_NAME, ACTIVATION, USE_BIAS, BIAS_CONSTRAINT,
                                BIAS_INITIALIZER, BIAS_REGULARIZER, TRAINABLE, ACTIVITY_REGULARIZER,
                                KERNEL_INITIALIZER, KERNEL_REGULARIZER, KERNEL_CONSTRAINT]

                dense_cfg = wrapped_layer.layer.get_config()
                pw_cfg = {attr: dense_cfg[attr] for attr in pw_attr_list}

                # Use more attributes that are not taken as are
                pw_cfg.update({KERNEL_SIZE: (1, 1),
                               STRIDES: (1, 1),
                               PADDING: PAD_VALID,
                               FILTERS: dense_cfg[UNITS]})
                # pw_cfg = {}
                pw_layer = Conv2D(**pw_cfg)
                pw_layer.build(wrapped_layer.layer.input_shape)


                dense_kernel = wrapped_layer.layer.kernel
                pw_weights = []
                pw_kernel = np.reshape(wrapped_layer.get_weights()[0], (1, 1, dense_kernel.get_shape()[0], dense_cfg[UNITS]))
                pw_weights.append(pw_kernel)
                if wrapped_layer.layer.use_bias:
                    pw_bias = wrapped_layer.get_weights()[2]
                    pw_weights.append(pw_bias)
                # pw_kernel = dense_kernel.reshape()
                pw_layer.set_weights(pw_weights)

                kernel_quantizer_cfg = wrapped_layer.dispatcher.weight_quantizers['kernel'].get_config()
                kernel_quantizer_cfg['channel_axis'] = 3
                kernel_quantizer_cfg['input_num_dims'] = 4
                quantizer_class = type(wrapped_layer.dispatcher.weight_quantizers['kernel'])
                new_quantizer = quantizer_class(**kernel_quantizer_cfg)

                new_weights_quantizer = copy.deepcopy(wrapped_layer.dispatcher.weight_quantizers)
                new_weights_quantizer['kernel'] = new_quantizer

                new_dispatcher = copy.deepcopy(wrapped_layer.dispatcher)
                new_dispatcher.set_weight_quantizers(new_weights_quantizer)

                dim = wrapped_layer.layer.input_shape[1:][:-1]

                target_shape = (1,int(np.prod(dim))) + (dense_kernel.get_shape()[0],)

                return Sequential([
                    Reshape(target_shape=target_shape),
                    qi.KerasQuantizationWrapper(pw_layer,
                                                new_dispatcher),
                    Reshape(wrapped_layer.layer.output_shape[1:])
                ])
            return wrapped_layer

        self.transformed_model = clone_model(self.model, clone_function=_substitute_model)
        converter = tf.lite.TFLiteConverter.from_keras_model(self.transformed_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.exported_model = converter.convert()

        Logger.info(f'Exporting INT8 tflite model to: {self.save_model_path}')
        with open(self.save_model_path, 'wb') as f:
            f.write(self.exported_model)


