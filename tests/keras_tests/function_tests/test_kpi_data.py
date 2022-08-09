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
from typing import List

from tensorflow import initializers

import model_compression_toolkit as mct
import numpy as np
import keras
import unittest
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Input, SeparableConv2D

from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs
from model_compression_toolkit.core.keras.constants import DEPTHWISE_KERNEL, KERNEL
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.separableconv_decomposition import \
    POINTWISE_KERNEL
from tests.common_tests.helpers.activation_mp_tp_model import generate_tp_model_with_activation_mp
from tests.keras_tests.tpc_keras import generate_activation_mp_tpc_keras


def small_random_datagen():
    return [np.random.random((1, 8, 8, 3))]


def large_random_datagen():
    return [np.random.random((1, 224, 224, 3))]


def compute_output_size(output_shape):
    output_shapes = output_shape if isinstance(output_shape, List) else [output_shape]
    output_shapes = [s[1:] for s in output_shapes]
    return sum([np.prod([x for x in output_shape if x is not None]) for output_shape in output_shapes])


def basic_model():
    random_uniform = initializers.random_uniform(0, 1)
    inputs = Input(shape=(8, 8, 3))
    x = Conv2D(2, 3, padding='same', name="conv2d")(inputs)
    x_bn = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                              moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                              name="bn1")(x)
    outputs = ReLU()(x_bn)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model, \
           getattr(model.layers[1], KERNEL).numpy().flatten().shape[0], \
           compute_output_size(model.layers[0].output_shape)


def complex_model():
    random_uniform = initializers.random_uniform(0, 1)
    inputs = Input(shape=(224, 224, 3))
    x = SeparableConv2D(10, 6, padding='same', name="sep_conv2d1")(inputs)
    x = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                           moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                           name="bn1")(x)
    x = ReLU()(x)
    x = SeparableConv2D(20, 12, padding='same', name="sep_conv2d2")(x)
    x = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                           moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                           name="bn2")(x)
    outputs = ReLU()(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model, \
           getattr(model.layers[1], DEPTHWISE_KERNEL).numpy().flatten().shape[0] + \
           getattr(model.layers[1], POINTWISE_KERNEL).numpy().flatten().shape[0] + \
           getattr(model.layers[4], DEPTHWISE_KERNEL).numpy().flatten().shape[0] + \
           getattr(model.layers[4], POINTWISE_KERNEL).numpy().flatten().shape[0], \
           compute_output_size(model.layers[4].output_shape)


def prep_test(model, mp_bitwidth_candidates_list, random_datagen):
    base_config, mixed_precision_cfg_list = get_op_quantization_configs()
    base_config = base_config.clone_and_edit(weights_n_bits=mp_bitwidth_candidates_list[0][0],
                                             activation_n_bits=mp_bitwidth_candidates_list[0][1])
    tp_model = generate_tp_model_with_activation_mp(
        base_cfg=base_config,
        mp_bitwidth_candidates_list=mp_bitwidth_candidates_list)
    tpc = generate_activation_mp_tpc_keras(tp_model=tp_model, name="kpi_data_test")

    kpi_data = mct.keras_kpi_data(in_model=model,
                                  representative_data_gen=random_datagen,
                                  target_platform_capabilities=tpc)

    return kpi_data


class TestKPIData(unittest.TestCase):

    def test_kpi_data_basic_all_bitwidth(self):
        model, sum_parameters, max_tensor = basic_model()
        mp_bitwidth_candidates_list = [(i, j) for i in [8, 4, 2] for j in [8, 4, 2]]

        kpi_data = prep_test(model, mp_bitwidth_candidates_list, small_random_datagen)

        self.verify_results(kpi_data, sum_parameters, max_tensor)

    def test_kpi_data_basic_partial_bitwidth(self):
        model, sum_parameters, max_tensor = basic_model()
        mp_bitwidth_candidates_list = [(i, j) for i in [4, 2] for j in [4, 2]]

        kpi_data = prep_test(model, mp_bitwidth_candidates_list, small_random_datagen)

        self.verify_results(kpi_data, sum_parameters, max_tensor)

    def test_kpi_data_complex_all_bitwidth(self):
        model, sum_parameters, max_tensor = complex_model()
        mp_bitwidth_candidates_list = [(i, j) for i in [8, 4, 2] for j in [8, 4, 2]]

        kpi_data = prep_test(model, mp_bitwidth_candidates_list, large_random_datagen())

        self.verify_results(kpi_data, sum_parameters, max_tensor)

    def test_kpi_data_complex_partial_bitwidth(self):
        model, sum_parameters, max_tensor = basic_model()
        mp_bitwidth_candidates_list = [(i, j) for i in [4, 2] for j in [4, 2]]

        kpi_data = prep_test(model, mp_bitwidth_candidates_list, small_random_datagen)

        self.verify_results(kpi_data, sum_parameters, max_tensor)

    def verify_results(self, kpi, sum_parameters, max_tensor):
        self.assertTrue(kpi.weights_memory == sum_parameters,
                        f"Expects weights_memory to be {sum_parameters} "
                        f"but result is {kpi.weights_memory}")
        self.assertTrue(kpi.activation_memory == max_tensor,
                        f"Expects activation_memory to be {max_tensor} "
                        f"but result is {kpi.activation_memory}")


if __name__ == '__main__':
    unittest.main()
