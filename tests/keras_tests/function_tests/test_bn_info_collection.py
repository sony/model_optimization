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
import keras
import unittest
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Input, SeparableConv2D, Reshape
from tensorflow import initializers
import numpy as np

from model_compression_toolkit import DEFAULTCONFIG
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute


def create_model_1(input_shape):
    random_uniform = initializers.random_uniform(0, 1)
    inputs = Input(shape=input_shape)
    x = Conv2D(2, 3, padding='same', name="conv2d")(inputs)
    x_bn = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                              moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                              name="bn1")(x)
    outputs = ReLU()(x_bn)
    return keras.Model(inputs=inputs, outputs=outputs)


def create_model_2(input_shape):
    random_uniform = initializers.random_uniform(0, 1)
    inputs = Input(shape=input_shape)
    x = SeparableConv2D(2, 3, padding='same', name="conv2d")(inputs)
    x_bn = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                              moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                              name="bn1")(x)
    outputs = ReLU()(x_bn)
    return keras.Model(inputs=inputs, outputs=outputs)


def create_model_3(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(2, 3, padding='same', name="conv2d")(inputs)
    x_bn = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                              name="bn1")(x)
    x_bn2 = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                               name="bn2")(x)
    x_relu1 = ReLU()(x_bn)
    x_relu2 = ReLU(max_value=6.)(x_bn2)
    outputs = x_relu1 + x_relu2
    return keras.Model(inputs=inputs, outputs=outputs)


def create_model_4(input_shape):
    random_uniform = initializers.random_uniform(0, 1)
    inputs = Input(shape=input_shape)
    x = Conv2D(2, 3, padding='same', name="conv2d")(inputs)
    x_bn = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                              moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                              name='bn1')(x)
    x_bn2 = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                               moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                               name='bn2')(x_bn)
    outputs = ReLU()(x_bn2)
    return keras.Model(inputs=inputs, outputs=outputs)


def create_model_5(input_shape):
    random_uniform = initializers.random_uniform(0, 1)
    inputs = Input(shape=input_shape, name='input1')
    x_bn = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                              moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                              name="bn1")(inputs)
    x_bn2 = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                               moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                               name='bn2')(x_bn)
    outputs = ReLU()(x_bn2)
    return keras.Model(inputs=inputs, outputs=outputs)


def create_model_6(input_shape):
    random_uniform = initializers.random_uniform(0, 1)
    inputs = Input(shape=input_shape, name='input1')
    x_bn = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                              moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                              name="bn1")(inputs)
    x_relu = ReLU(name='relu1')(x_bn)
    x_bn2 = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                               moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                               name='bn2')(x_relu)
    x_reshape = Reshape((-1,), name='reshape1')(x_bn2)
    x_bn3 = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                               moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                               name='bn3')(
        x_reshape)
    outputs = ReLU(name='relu2')(x_bn3)
    return keras.Model(inputs=inputs, outputs=outputs)


def create_model_7(input_shape):
    inputs = Input(shape=input_shape, name='input1')
    x_bn = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                              name="bn1")(inputs)
    x_bn2 = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                               name='bn2')(inputs)
    x_relu = ReLU()(x_bn2)
    outputs = x_bn + x_relu
    return keras.Model(inputs=inputs, outputs=outputs)


def create_model_8(input_shape):
    inputs = Input(shape=input_shape, name='input1')
    x_bn = BatchNormalization(name="bn1", scale=False, center=False)(inputs)
    outputs = x_bn
    return keras.Model(inputs=inputs, outputs=outputs)


def prepare_graph(in_model):
    fw_info = DEFAULT_KERAS_INFO
    keras_impl = KerasImplementation()

    def dummy_representative_dataset():
        return None

    graph = keras_impl.model_reader(in_model, dummy_representative_dataset)  # model reading
    graph = substitute(graph, keras_impl.get_substitutions_prepare_graph())
    for node in graph.nodes:
        node.prior_info = keras_impl.get_node_prior_info(node=node,
                                                         fw_info=fw_info, graph=graph)
    transformed_graph = substitute(graph, keras_impl.get_substitutions_pre_statistics_collection(DEFAULTCONFIG))
    return transformed_graph


class TestBNInfoCollection(unittest.TestCase):

    def test_conv2d_bn_info_collection(self):
        input_shape = (8, 8, 3)
        in_model = create_model_1(input_shape)
        transformed_graph = prepare_graph(in_model)

        self.assertTrue(len(transformed_graph.find_node_by_name('conv2d_bn')) == 1)
        conv_bn_node = transformed_graph.find_node_by_name('conv2d_bn')[0]
        prior_std = conv_bn_node.prior_info.std_output
        prior_mean = conv_bn_node.prior_info.mean_output

        self.assertTrue(not (prior_std is None))
        self.assertTrue(not (prior_mean is None))

        bn_layer = in_model.get_layer("bn1")
        gamma = bn_layer.gamma
        beta = bn_layer.beta
        self.assertTrue((abs(gamma) == prior_std).numpy().all())
        self.assertTrue((beta == prior_mean).numpy().all())

    def test_seperableconv2d_bn_info_collection(self):
        input_shape = (8, 8, 3)
        in_model = create_model_2(input_shape)
        transformed_graph = prepare_graph(in_model)

        self.assertTrue(len(transformed_graph.find_node_by_name('conv2d_pw_bn')) == 1)
        conv_bn_node = transformed_graph.find_node_by_name('conv2d_pw_bn')[0]
        prior_std = conv_bn_node.prior_info.std_output
        prior_mean = conv_bn_node.prior_info.mean_output

        self.assertTrue(not (prior_std is None))
        self.assertTrue(not (prior_mean is None))

        bn_layer = in_model.get_layer("bn1")
        gamma = bn_layer.gamma
        beta = bn_layer.beta
        self.assertTrue((abs(gamma) == prior_std).numpy().all())
        self.assertTrue((beta == prior_mean).numpy().all())

    def test_conv2d_2bn_info_collection(self):
        input_shape = (8, 8, 3)
        in_model = create_model_3(input_shape)
        transformed_graph = prepare_graph(in_model)

        self.assertTrue(len(transformed_graph.find_node_by_name('conv2d')) == 1)
        conv_node = transformed_graph.find_node_by_name('conv2d')[0]

        self.assertTrue(len(transformed_graph.find_node_by_name('bn1')) == 1)
        bn_node = transformed_graph.find_node_by_name('bn1')[0]

        self.assertTrue(len(transformed_graph.find_node_by_name('bn2')) == 1)
        bn2_node = transformed_graph.find_node_by_name('bn2')[0]

        conv_std = conv_node.prior_info.std_output
        conv_mean = conv_node.prior_info.mean_output

        self.assertTrue(not (conv_std is None))
        self.assertTrue(not (conv_mean is None))

        bn_std = bn_node.prior_info.std_output
        bn_mean = bn_node.prior_info.mean_output

        self.assertTrue(not (bn_std is None))
        self.assertTrue(not (bn_mean is None))

        bn2_std = bn2_node.prior_info.std_output
        bn2_mean = bn2_node.prior_info.mean_output

        self.assertTrue(not (bn2_std is None))
        self.assertTrue(not (bn2_mean is None))

        bn_layer = in_model.get_layer("bn1")
        mm = bn_layer.moving_mean
        mv = bn_layer.moving_variance
        m_std = np.sqrt(mv)
        self.assertTrue((m_std == conv_std).all())
        self.assertTrue((mm == conv_mean).numpy().all())

        gamma = bn_layer.gamma
        beta = bn_layer.beta
        self.assertTrue((abs(gamma) == bn_std).numpy().all())
        self.assertTrue((beta == bn_mean).numpy().all())

        bn2_layer = in_model.get_layer("bn2")
        gamma2 = bn2_layer.gamma
        beta2 = bn2_layer.beta
        self.assertTrue((abs(gamma2) == bn2_std).numpy().all())
        self.assertTrue((beta2 == bn2_mean).numpy().all())

    def test_conv2d_bn_chain_info_collection(self):
        input_shape = (8, 8, 3)
        in_model = create_model_4(input_shape)
        transformed_graph = prepare_graph(in_model)

        self.assertTrue(len(transformed_graph.find_node_by_name('conv2d_bn')) == 1)
        conv_bn_node = transformed_graph.find_node_by_name('conv2d_bn')[0]

        self.assertTrue(len(transformed_graph.find_node_by_name('bn2')) == 1)
        bn2_node = transformed_graph.find_node_by_name('bn2')[0]

        conv_std = conv_bn_node.prior_info.std_output
        conv_mean = conv_bn_node.prior_info.mean_output

        self.assertTrue(not (conv_std is None))
        self.assertTrue(not (conv_mean is None))

        bn_std = bn2_node.prior_info.std_output
        bn_mean = bn2_node.prior_info.mean_output

        self.assertTrue(not (bn_std is None))
        self.assertTrue(not (bn_mean is None))

        bn_layer = in_model.get_layer("bn1")
        gamma = bn_layer.gamma
        beta = bn_layer.beta
        self.assertTrue((abs(gamma) == conv_std).numpy().all())
        self.assertTrue((beta == conv_mean).numpy().all())

        bn2_layer = in_model.get_layer("bn2")
        gamma2 = bn2_layer.gamma
        beta2 = bn2_layer.beta
        self.assertTrue((abs(gamma2) == bn_std).numpy().all())
        self.assertTrue((beta2 == bn_mean).numpy().all())

    def test_bn_chain_info_collection(self):
        input_shape = (8, 8, 3)
        in_model = create_model_5(input_shape)
        transformed_graph = prepare_graph(in_model)

        self.assertTrue(len(transformed_graph.find_node_by_name('input1')) == 1)
        input_node = transformed_graph.find_node_by_name('input1')[0]

        self.assertTrue(len(transformed_graph.find_node_by_name('bn1')) == 1)
        bn_node = transformed_graph.find_node_by_name('bn1')[0]

        self.assertTrue(len(transformed_graph.find_node_by_name('bn2')) == 1)
        bn2_node = transformed_graph.find_node_by_name('bn2')[0]

        input_std = input_node.prior_info.std_output
        input_mean = input_node.prior_info.mean_output

        self.assertTrue(not (input_std is None))
        self.assertTrue(not (input_mean is None))

        bn_std = bn_node.prior_info.std_output
        bn_mean = bn_node.prior_info.mean_output

        self.assertTrue(not (bn_std is None))
        self.assertTrue(not (bn_mean is None))

        bn2_std = bn2_node.prior_info.std_output
        bn2_mean = bn2_node.prior_info.mean_output

        self.assertTrue(not (bn2_std is None))
        self.assertTrue(not (bn2_mean is None))

        bn_layer = in_model.get_layer("bn1")
        mm = bn_layer.moving_mean
        mv = bn_layer.moving_variance
        m_std = np.sqrt(mv)
        self.assertTrue((m_std == input_std).all())
        self.assertTrue((mm == input_mean).numpy().all())

        gamma = bn_layer.gamma
        beta = bn_layer.beta
        self.assertTrue((abs(gamma) == bn_std).numpy().all())
        self.assertTrue((beta == bn_mean).numpy().all())

        bn2_layer = in_model.get_layer("bn2")
        gamma2 = bn2_layer.gamma
        beta2 = bn2_layer.beta
        self.assertTrue((abs(gamma2) == bn2_std).numpy().all())
        self.assertTrue((beta2 == bn2_mean).numpy().all())

    def test_layers_bn_info_collection(self):
        input_shape = (8, 8, 3)
        in_model = create_model_6(input_shape)
        transformed_graph = prepare_graph(in_model)

        self.assertTrue(len(transformed_graph.find_node_by_name('input1')) == 1)
        input_node = transformed_graph.find_node_by_name('input1')[0]

        self.assertTrue(len(transformed_graph.find_node_by_name('bn1')) == 1)
        bn_node = transformed_graph.find_node_by_name('bn1')[0]

        self.assertTrue(len(transformed_graph.find_node_by_name('relu1')) == 1)
        relu_node = transformed_graph.find_node_by_name('relu1')[0]

        self.assertTrue(len(transformed_graph.find_node_by_name('bn2')) == 1)
        bn2_node = transformed_graph.find_node_by_name('bn2')[0]

        self.assertTrue(len(transformed_graph.find_node_by_name('reshape1')) == 1)
        reshape_node = transformed_graph.find_node_by_name('reshape1')[0]

        self.assertTrue(len(transformed_graph.find_node_by_name('bn3')) == 1)
        bn3_node = transformed_graph.find_node_by_name('bn3')[0]

        input_std = input_node.prior_info.std_output
        input_mean = input_node.prior_info.mean_output

        self.assertTrue(not (input_std is None))
        self.assertTrue(not (input_mean is None))

        bn_std = bn_node.prior_info.std_output
        bn_mean = bn_node.prior_info.mean_output

        self.assertTrue(not (bn_std is None))
        self.assertTrue(not (bn_mean is None))

        relu_std = relu_node.prior_info.std_output
        relu_mean = relu_node.prior_info.mean_output

        self.assertTrue(not (relu_std is None))
        self.assertTrue(not (relu_mean is None))

        bn2_std = bn2_node.prior_info.std_output
        bn2_mean = bn2_node.prior_info.mean_output

        self.assertTrue(not (bn2_std is None))
        self.assertTrue(not (bn2_mean is None))

        reshape_std = reshape_node.prior_info.std_output
        reshape_mean = reshape_node.prior_info.mean_output

        self.assertTrue(not (reshape_std is None))
        self.assertTrue(not (reshape_mean is None))

        bn3_std = bn3_node.prior_info.std_output
        bn3_mean = bn3_node.prior_info.mean_output

        self.assertTrue(not (bn3_std is None))
        self.assertTrue(not (bn3_mean is None))

        bn_layer = in_model.get_layer("bn1")
        mm = bn_layer.moving_mean
        mv = bn_layer.moving_variance
        m_std = np.sqrt(mv)
        self.assertTrue((m_std == input_std).all())
        self.assertTrue((mm == input_mean).numpy().all())

        gamma = bn_layer.gamma
        beta = bn_layer.beta
        self.assertTrue((abs(gamma) == bn_std).numpy().all())
        self.assertTrue((beta == bn_mean).numpy().all())

        bn2_layer = in_model.get_layer("bn2")
        mm2 = bn2_layer.moving_mean
        mv2 = bn2_layer.moving_variance
        m_std2 = np.sqrt(mv2)
        self.assertTrue((m_std2 == relu_std).all())
        self.assertTrue((mm2 == relu_mean).numpy().all())

        gamma2 = bn2_layer.gamma
        beta2 = bn2_layer.beta
        self.assertTrue((abs(gamma2) == bn2_std).numpy().all())
        self.assertTrue((beta2 == bn2_mean).numpy().all())

        bn3_layer = in_model.get_layer("bn3")
        mm3 = bn3_layer.moving_mean
        mv3 = bn3_layer.moving_variance
        m_std3 = np.sqrt(mv3)
        self.assertTrue((m_std3 == reshape_std).all())
        self.assertTrue((mm3 == reshape_mean).numpy().all())

        gamma3 = bn3_layer.gamma
        beta3 = bn3_layer.beta
        self.assertTrue((abs(gamma3) == bn3_std).numpy().all())
        self.assertTrue((beta3 == bn3_mean).numpy().all())

    def test_inp_2bn_info_collection(self):
        input_shape = (8, 8, 3)
        in_model = create_model_7(input_shape)
        transformed_graph = prepare_graph(in_model)

        self.assertTrue(len(transformed_graph.find_node_by_name('input1')) == 1)
        input_node = transformed_graph.find_node_by_name('input1')[0]

        self.assertTrue(len(transformed_graph.find_node_by_name('bn1')) == 1)
        bn_node = transformed_graph.find_node_by_name('bn1')[0]

        self.assertTrue(len(transformed_graph.find_node_by_name('bn2')) == 1)
        bn2_node = transformed_graph.find_node_by_name('bn2')[0]

        input_std = input_node.prior_info.std_output
        input_mean = input_node.prior_info.mean_output

        self.assertTrue(not (input_std is None))
        self.assertTrue(not (input_mean is None))

        bn_std = bn_node.prior_info.std_output
        bn_mean = bn_node.prior_info.mean_output

        self.assertTrue(not (bn_std is None))
        self.assertTrue(not (bn_mean is None))

        bn2_std = bn2_node.prior_info.std_output
        bn2_mean = bn2_node.prior_info.mean_output

        self.assertTrue(not (bn2_std is None))
        self.assertTrue(not (bn2_mean is None))

        bn_layer = in_model.get_layer("bn1")
        mm = bn_layer.moving_mean
        mv = bn_layer.moving_variance
        m_std = np.sqrt(mv)
        self.assertTrue((m_std == input_std).all())
        self.assertTrue((mm == input_mean).numpy().all())

        gamma = bn_layer.gamma
        beta = bn_layer.beta
        self.assertTrue((abs(gamma) == bn_std).numpy().all())
        self.assertTrue((beta == bn_mean).numpy().all())

        bn2_layer = in_model.get_layer("bn2")
        gamma2 = bn2_layer.gamma
        beta2 = bn2_layer.beta
        self.assertTrue((abs(gamma2) == bn2_std).numpy().all())
        self.assertTrue((beta2 == bn2_mean).numpy().all())

    def test_no_scale_center_bn_info_collection(self):
        input_shape = (8, 8, 3)
        in_model = create_model_8(input_shape)
        transformed_graph = prepare_graph(in_model)

        self.assertTrue(len(transformed_graph.find_node_by_name('input1')) == 1)
        input_node = transformed_graph.find_node_by_name('input1')[0]

        self.assertTrue(len(transformed_graph.find_node_by_name('bn1')) == 1)
        bn_node = transformed_graph.find_node_by_name('bn1')[0]

        input_std = input_node.prior_info.std_output
        input_mean = input_node.prior_info.mean_output

        self.assertTrue(not (input_std is None))
        self.assertTrue(not (input_mean is None))

        bn_std = bn_node.prior_info.std_output
        bn_mean = bn_node.prior_info.mean_output

        self.assertTrue(not (bn_std is None))
        self.assertTrue(not (bn_mean is None))

        bn_layer = in_model.get_layer("bn1")
        mm = bn_layer.moving_mean
        mv = bn_layer.moving_variance
        m_std = np.sqrt(mv)
        self.assertTrue((m_std == input_std).all())
        self.assertTrue((mm == input_mean).numpy().all())

        gamma = bn_layer.gamma
        beta = bn_layer.beta
        if gamma is None:
            gamma = 1.0
        if beta is None:
            beta = 0.0
        self.assertTrue((abs(gamma) == bn_std))
        self.assertTrue((beta == bn_mean))


if __name__ == '__main__':
    unittest.main()
