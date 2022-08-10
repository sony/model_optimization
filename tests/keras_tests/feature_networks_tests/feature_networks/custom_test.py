# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.engine.base_layer import InputSpec
from keras.layers import Conv2D
from keras.models import save_model, load_model
from keras.utils import conv_utils

from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest

keras = tf.keras
layers = keras.layers

import tensorflow_model_optimization as tfmot


class MyLayer(Conv2D):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MyLayer, self).build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                'The number of input channels must be evenly divisible by the number '
                'of groups. Received groups={}, but the input has {} channels '
                '(full input shape is {}).'.format(self.groups, input_channel,
                                                   input_shape))
        kernel_shape = self.kernel_size + (input_channel // self.groups,
                                           self.filters)

        # compute_output_shape contains some validation logic for the input shape,
        # and make sure the output shape has all positive dimensions.
        self.compute_output_shape(input_shape)

        self.kernel = tf.Variable(np.random.randint(-100, 100, size=kernel_shape, dtype=np.int8),
                                  trainable=False,
                                  dtype=tf.int8)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={channel_axis: input_channel})
        self.built = True

    def call(self, inputs):
        # return super(MyLayer, self).call(inputs)
        input_shape = inputs.shape

        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = tf.pad(inputs, self._compute_causal_padding(inputs))

        outputs = self.convolution_op(inputs, tf.cast(self.kernel, tf.float32))
        # outputs = self.convolution_op(inputs, self.kernel)

        if self.use_bias:
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first:
                # nn.bias_add does not accept a 1D input tensor.
                bias = tf.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            else:
                # Handle multiple batch dimensions.
                if output_rank is not None and output_rank > 2 + self.rank:

                    def _apply_fn(o):
                        return tf.nn.bias_add(o, self.bias, data_format=self._tf_data_format)

                    outputs = conv_utils.squeeze_batch_dims(
                        outputs, _apply_fn, inner_rank=self.rank + 1)
                else:
                    outputs = tf.nn.bias_add(
                        outputs, self.bias, data_format=self._tf_data_format)

        if not tf.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(input_shape)
            outputs.set_shape(out_shape)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        cfg = super(MyLayer, self).get_config()
        # cfg.update({})
        return cfg


class CustomLayerTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, input_shape=(224, 224, 3))

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        outputs = MyLayer(filters=30, kernel_size=9)(inputs)
        for _ in range(10):
            outputs = MyLayer(filters=30, kernel_size=9)(outputs)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        custom_objects = {"MyLayer": MyLayer}
        save_model(quantized_model, './qmodel.h5')
        with tfmot.quantization.keras.quantize_scope():
            quantized_model = load_model('./qmodel.h5', custom_objects)
        quantized_model(input_x)
        converter = tf.lite.TFLiteConverter.from_keras_model(quantized_model)
        # converter.experimental_new_converter = not args.disable_experimental_new_converter
        tflite_quant_model = converter.convert()
        # model_name = model.name.replace('.', '_')
        # model_filename = f'{model_name}_{FILE_TIME_STAMP}_debug.tflite' if args.run_debugger else f'{model_name}_' \
        #                                                                                           f'{
        # FILE_TIME_STAMP}.tflite'
        model_file = './qmodel.tflite'
        Path(model_file).write_bytes(tflite_quant_model)
        print(f"TFLite Model: {model_file}")
