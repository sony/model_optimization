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

import functools
import keras
import numpy as np
import unittest
from keras.layers import Dense
from tensorflow import initializers
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Input, Conv2DTranspose, DepthwiseConv2D

import model_compression_toolkit as mct
import model_compression_toolkit.core.common.hessian as hessian_common
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_configs

tp = mct.target_platform


def basic_model(input_shape, layer):
    random_uniform = initializers.random_uniform(0, 1)
    inputs = Input(shape=input_shape[1:])
    x = layer(inputs)
    x_bn = BatchNormalization(gamma_initializer='random_normal', beta_initializer='random_normal',
                              moving_mean_initializer='random_normal', moving_variance_initializer=random_uniform,
                              name="bn1")(x)
    outputs = ReLU()(x_bn)
    return keras.Model(inputs=inputs, outputs=outputs)


def representative_dataset(input_shape):
    yield [np.random.randn(*input_shape).astype(np.float32)]


class TestHessianInfoCalculatorWeights(unittest.TestCase):

    def _fetch_scores(self, hessian_info, target_node, granularity, num_scores=1):
        request = hessian_common.TraceHessianRequest(mode=hessian_common.HessianMode.WEIGHTS,
                                                     granularity=granularity,
                                                     target_node=target_node)
        info = hessian_info.fetch_hessian(request, num_scores)
        assert len(info) == num_scores, f"fetched {num_scores} score but {len(info)} scores were fetched"
        return np.mean(np.stack(info), axis=0)

    def _test_score_shape(self, hessian_service, interest_point, granularity, expected_shape, num_scores=1):
        score = self._fetch_scores(hessian_info=hessian_service,
                                   target_node=interest_point,  # linear op
                                   granularity=granularity,
                                   num_scores=num_scores)
        self.assertTrue(isinstance(score, np.ndarray), f"scores expected to be a numpy array but is {type(score)}")
        self.assertTrue(score.shape == expected_shape,
                        f"Tensor shape is expected to be {expected_shape} but has shape {score.shape}")  # per tensor
        return score

    def test_conv2d_granularity(self):
        input_shape = (1, 8, 8, 3)
        in_model = basic_model(input_shape, layer=Conv2D(filters=2, kernel_size=3))
        keras_impl = KerasImplementation()
        _repr_dataset = functools.partial(representative_dataset,
                                          input_shape=input_shape)
        graph = prepare_graph_with_configs(in_model,
                                           keras_impl,
                                           DEFAULT_KERAS_INFO,
                                           _repr_dataset,
                                           generate_keras_tpc)

        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        interest_points = [n for n in sorted_graph_nodes]
        hessian_service = hessian_common.HessianInfoService(graph=graph,
                                                            representative_dataset=_repr_dataset,
                                                            fw_impl=keras_impl)
        self._test_score_shape(hessian_service,
                               interest_points[1],
                               granularity=hessian_common.HessianInfoGranularity.PER_TENSOR,
                               expected_shape=(1,))
        self._test_score_shape(hessian_service,
                               interest_points[1],
                               granularity=hessian_common.HessianInfoGranularity.PER_OUTPUT_CHANNEL,
                               expected_shape=(2,))
        self._test_score_shape(hessian_service,
                               interest_points[1],
                               granularity=hessian_common.HessianInfoGranularity.PER_ELEMENT,
                               expected_shape=(3, 3, 3, 2))

    def test_dense_granularity(self):
        input_shape = (1, 8)
        in_model = basic_model(input_shape, layer=Dense(2))
        keras_impl = KerasImplementation()
        _repr_dataset = functools.partial(representative_dataset,
                                          input_shape=input_shape)
        graph = prepare_graph_with_configs(in_model,
                                           keras_impl,
                                           DEFAULT_KERAS_INFO,
                                           _repr_dataset,
                                           generate_keras_tpc)

        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        interest_points = [n for n in sorted_graph_nodes]
        hessian_service = hessian_common.HessianInfoService(graph=graph,
                                                            representative_dataset=_repr_dataset,
                                                            fw_impl=keras_impl)

        self._test_score_shape(hessian_service,
                               interest_points[1],
                               granularity=hessian_common.HessianInfoGranularity.PER_TENSOR,
                               expected_shape=(1,))
        self._test_score_shape(hessian_service,
                               interest_points[1],
                               granularity=hessian_common.HessianInfoGranularity.PER_OUTPUT_CHANNEL,
                               expected_shape=(2,))
        self._test_score_shape(hessian_service,
                               interest_points[1],
                               granularity=hessian_common.HessianInfoGranularity.PER_ELEMENT,
                               expected_shape=(8, 2))

    def test_conv2dtranspose_granularity(self):
        input_shape = (1, 8, 8, 3)
        in_model = basic_model(input_shape, layer=Conv2DTranspose(filters=2, kernel_size=3))
        keras_impl = KerasImplementation()
        _repr_dataset = functools.partial(representative_dataset,
                                          input_shape=input_shape)
        graph = prepare_graph_with_configs(in_model,
                                           keras_impl,
                                           DEFAULT_KERAS_INFO,
                                           _repr_dataset,
                                           generate_keras_tpc)

        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        interest_points = [n for n in sorted_graph_nodes]
        hessian_service = hessian_common.HessianInfoService(graph=graph,
                                                            representative_dataset=_repr_dataset,
                                                            fw_impl=keras_impl)

        self._test_score_shape(hessian_service,
                               interest_points[1],
                               granularity=hessian_common.HessianInfoGranularity.PER_TENSOR,
                               expected_shape=(1,))
        self._test_score_shape(hessian_service,
                               interest_points[1],
                               granularity=hessian_common.HessianInfoGranularity.PER_OUTPUT_CHANNEL,
                               expected_shape=(2,))
        self._test_score_shape(hessian_service,
                               interest_points[1],
                               granularity=hessian_common.HessianInfoGranularity.PER_ELEMENT,
                               expected_shape=(3, 3, 2, 3))

    def test_depthwiseconv2d_granularity(self):
        input_shape = (1, 8, 8, 3)
        in_model = basic_model(input_shape, layer=DepthwiseConv2D(kernel_size=3))
        keras_impl = KerasImplementation()
        _repr_dataset = functools.partial(representative_dataset,
                                          input_shape=input_shape)
        graph = prepare_graph_with_configs(in_model,
                                           keras_impl,
                                           DEFAULT_KERAS_INFO,
                                           _repr_dataset,
                                           generate_keras_tpc)

        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        interest_points = [n for n in sorted_graph_nodes]
        hessian_service = hessian_common.HessianInfoService(graph=graph,
                                                            representative_dataset=_repr_dataset,
                                                            fw_impl=keras_impl)

        self._test_score_shape(hessian_service,
                               interest_points[1],
                               granularity=hessian_common.HessianInfoGranularity.PER_TENSOR,
                               expected_shape=(1,))
        self._test_score_shape(hessian_service,
                               interest_points[1],
                               granularity=hessian_common.HessianInfoGranularity.PER_OUTPUT_CHANNEL,
                               expected_shape=(3,))
        self._test_score_shape(hessian_service,
                               interest_points[1],
                               granularity=hessian_common.HessianInfoGranularity.PER_ELEMENT,
                               expected_shape=(3, 3, 3, 1))

