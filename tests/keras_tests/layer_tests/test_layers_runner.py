# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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

import unittest
from functools import partial

import tensorflow as tf
from keras.layers import Softmax, Activation, LeakyReLU, ReLU, ZeroPadding2D, UpSampling2D, Multiply, PReLU, Reshape, \
    MaxPooling2D, Add, GlobalAveragePooling2D, Flatten, Dropout, Dot, DepthwiseConv2D, Dense, Cropping2D, \
    Conv2DTranspose, Conv2D, Concatenate, AveragePooling2D

from tests.keras_tests.layer_tests.base_keras_layer_test import BaseKerasLayerTest


class LayerTest(unittest.TestCase):

    def test_activation(self):
        BaseKerasLayerTest(self,
                           [Activation('linear'),
                            Activation('hard_sigmoid'),
                            Activation('exponential')]).run_test()

    def test_softplus(self):
        BaseKerasLayerTest(self,
                           [Activation('softplus'),
                            tf.nn.softplus]).run_test()

    def test_softsign(self):
        BaseKerasLayerTest(self,
                           [Activation('softsign'),
                            tf.nn.softsign]).run_test()

    def test_tanh(self):
        BaseKerasLayerTest(self,
                           [Activation('tanh'),
                            tf.nn.tanh]).run_test()

    def test_gelu(self):
        BaseKerasLayerTest(self,
                           [Activation('gelu'),
                            tf.nn.gelu,
                            partial(tf.nn.gelu, approximate=True)]).run_test()

    def test_selu(self):
        BaseKerasLayerTest(self,
                           [Activation('selu'),
                            tf.nn.selu]).run_test()

    def test_elu(self):
        BaseKerasLayerTest(self,
                           [Activation('elu'),
                            tf.nn.elu]).run_test()

    def test_leaky_relu(self):
        BaseKerasLayerTest(self,
                           [tf.nn.leaky_relu,
                            partial(tf.nn.leaky_relu, alpha=0.5),
                            LeakyReLU(),
                            LeakyReLU(alpha=0.6)]).run_test()

    def test_relu(self):
        BaseKerasLayerTest(self,
                           [Activation('relu'),
                            tf.nn.relu,
                            tf.nn.relu6,
                            ReLU(),
                            ReLU(max_value=6, negative_slope=0.001, threshold=1),
                            ReLU(4, 0.001, threshold=0.5),
                            ReLU(2.7)]).run_test()

    def test_swish(self):
        BaseKerasLayerTest(self,
                           [tf.nn.swish,
                            tf.nn.silu,
                            Activation('swish')]).run_test()

    def test_softmax(self):
        BaseKerasLayerTest(self,
                           [Activation('softmax'),
                            tf.nn.softmax,
                            partial(tf.nn.softmax, axis=1),
                            Softmax(),
                            Softmax(axis=2)]).run_test()

    def test_sigmoid(self):
        BaseKerasLayerTest(self,
                           [Activation('sigmoid'),
                            tf.nn.sigmoid]).run_test()

    def test_zeropadding2d(self):
        BaseKerasLayerTest(self,
                           [ZeroPadding2D(),
                            ZeroPadding2D(1),
                            ZeroPadding2D((3, 4))]).run_test()

    def test_upsampling2d(self):
        BaseKerasLayerTest(self,
                           [UpSampling2D(),
                            UpSampling2D(size=2),
                            UpSampling2D(size=(2, 1)),
                            UpSampling2D(interpolation='bilinear')]).run_test()

    def test_split(self):
        BaseKerasLayerTest(self,
                           [partial(tf.split, num_or_size_splits=1),
                            partial(tf.split, num_or_size_splits=2, axis=1)]).run_test()

    def test_resize(self):
        BaseKerasLayerTest(self,
                           [partial(tf.image.resize, size=[10, 20]),
                            partial(tf.image.resize, size=[10, 19], preserve_aspect_ratio=False),
                            partial(tf.image.resize, size=[9, 20], preserve_aspect_ratio=True),
                            partial(tf.image.resize, size=[9, 22],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)]).run_test()

    def test_reshape(self):
        BaseKerasLayerTest(self,
                           [Reshape((-1,)),
                            Reshape(target_shape=(-1,)),
                            Reshape(target_shape=(8, 12, 2)),
                            Reshape(target_shape=(64, 3, 1)),
                            partial(tf.reshape, shape=(-1,)),
                            partial(tf.reshape, shape=(8, 12, 2)),
                            partial(tf.reshape, shape=(64, 3, 1))
                            ]).run_test()

    def test_reduce_sum(self):
        BaseKerasLayerTest(self,
                           [partial(tf.reduce_sum, axis=0),
                            partial(tf.reduce_sum, axis=1),
                            partial(tf.reduce_sum, axis=0, keepdims=True),
                            partial(tf.reduce_sum, axis=0, keepdims=False)]).run_test()

    def test_reduce_min(self):
        BaseKerasLayerTest(self,
                           [partial(tf.reduce_min, axis=0),
                            partial(tf.reduce_min, axis=1),
                            partial(tf.reduce_min, axis=0, keepdims=True),
                            partial(tf.reduce_min, axis=0, keepdims=False)]).run_test()

    def test_reduce_max(self):
        BaseKerasLayerTest(self,
                           [partial(tf.reduce_max, axis=0),
                            partial(tf.reduce_max, axis=1),
                            partial(tf.reduce_max, axis=0, keepdims=True),
                            partial(tf.reduce_max, axis=0, keepdims=False)]).run_test()

    def test_reduce_mean(self):
        BaseKerasLayerTest(self,
                           [partial(tf.reduce_mean, axis=0),
                            partial(tf.reduce_mean, axis=1),
                            partial(tf.reduce_mean, axis=0, keepdims=True),
                            partial(tf.reduce_mean, axis=0, keepdims=False)]).run_test()

    def test_prelu(self):
        BaseKerasLayerTest(self,
                           [PReLU(),
                            PReLU(alpha_initializer='lecun_normalV2'),
                            PReLU(shared_axes=[0, 1]),
                            PReLU(alpha_regularizer=tf.keras.regularizers.L1(0.01))]).run_test()

    def test_multiply(self):
        BaseKerasLayerTest(self,
                           [Multiply()],
                           num_of_inputs=3,
                           is_inputs_a_list=True).run_test()

    def test_math_multiply(self):
        BaseKerasLayerTest(self,
                           [tf.multiply,
                            partial(tf.multiply, name="mul_test")],
                           num_of_inputs=2).run_test()

    def test_maxpooling2d(self):
        BaseKerasLayerTest(self,
                           [MaxPooling2D(),
                            MaxPooling2D(pool_size=(1, 2)),
                            MaxPooling2D(strides=2),
                            MaxPooling2D(padding='same')]).run_test()

    def test_add(self):
        BaseKerasLayerTest(self,
                           [Add()],
                           num_of_inputs=3,
                           is_inputs_a_list=True).run_test()

    def test_math_add(self):
        BaseKerasLayerTest(self,
                           [tf.add,
                            partial(tf.add, name="add_test")],
                           num_of_inputs=2).run_test()

    def test_globalaveragepooling2d(self):
        BaseKerasLayerTest(self,
                           [GlobalAveragePooling2D(),
                            GlobalAveragePooling2D(keepdims=True)]).run_test()

    def test_flatten(self):
        BaseKerasLayerTest(self,
                           [Flatten()]).run_test()

    def test_dropout(self):
        BaseKerasLayerTest(self,
                           [Dropout(rate=0.2),
                            Dropout(0.3, noise_shape=(8, 2, 3)),
                            Dropout(0.3, noise_shape=(8, 2, 3), seed=2)]).run_test()

    def test_dot(self):
        BaseKerasLayerTest(self,
                           [Dot(axes=1),
                            Dot(axes=[2, 1]),
                            Dot(axes=1, normalize=True)],
                           is_inputs_a_list=True,
                           num_of_inputs=2).run_test()

    def test_depthwiseConv2DTest(self):
        BaseKerasLayerTest(self,
                           [DepthwiseConv2D(1),
                            DepthwiseConv2D(1, depth_multiplier=3)
                            ]).run_test()

    def test_dense(self):
        BaseKerasLayerTest(self,
                           [Dense(2),
                            Dense(1, use_bias=False),
                            Dense(1, kernel_initializer='he_uniformV2')]).run_test()

    def test_cropping2d(self):
        BaseKerasLayerTest(self,
                           [Cropping2D(cropping=((1, 2), (2, 2))),
                            Cropping2D(),
                            Cropping2D(2)]).run_test()

    def test_crop_and_resize(self):
        boxes = tf.random.uniform(shape=(5, 4))
        box_indices = tf.random.uniform(shape=(5,), minval=0,
                                        maxval=1, dtype=tf.int32)
        BaseKerasLayerTest(self,
                           [partial(tf.image.crop_and_resize, boxes=boxes, box_indices=box_indices, crop_size=(22, 19)),
                            partial(tf.image.crop_and_resize, boxes=boxes, box_indices=box_indices, crop_size=(21, 24),
                                    method='nearest'),
                            partial(tf.image.crop_and_resize, boxes=boxes, box_indices=box_indices, crop_size=(24, 20),
                                    extrapolation_value=0)]).run_test()

    def test_conv2dtranspose(self):
        BaseKerasLayerTest(self,
                           [Conv2DTranspose(1, 1)
                            ],
                           use_cpu=True).run_test()  # Use CPU for inference as it seems to be non-deterministic on GPU

    def test_conv2d(self):
        BaseKerasLayerTest(self,
                           [Conv2D(1, 1),
                            Conv2D(1, 1, strides=2),
                            Conv2D(1, 1, use_bias=False)
                            ]).run_test()

    def test_concatenate(self):
        BaseKerasLayerTest(self,
                           [Concatenate(),
                            Concatenate(axis=0),
                            Concatenate(axis=1),
                            partial(tf.concat, axis=0),
                            partial(tf.concat, axis=1)
                            ],
                           is_inputs_a_list=True,
                           num_of_inputs=3).run_test()

    def test_averagepooling2d(self):
        BaseKerasLayerTest(self,
                           [AveragePooling2D(),
                            AveragePooling2D(pool_size=2),
                            AveragePooling2D(pool_size=(2, 1)),
                            AveragePooling2D(padding='same'),
                            AveragePooling2D(strides=2),
                            AveragePooling2D(strides=(2, 1))]).run_test()


if __name__ == '__main__':
    unittest.main()
