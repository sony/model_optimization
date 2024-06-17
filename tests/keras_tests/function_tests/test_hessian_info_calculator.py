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
import tensorflow as tf
import unittest
from keras.layers import Dense
from tensorflow import initializers
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Input, Conv2DTranspose, DepthwiseConv2D

import model_compression_toolkit as mct
import model_compression_toolkit.core.common.hessian as hessian_common
from model_compression_toolkit.core.keras.constants import KERNEL
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


def reused_model(input_shape):
    reused_layer = Conv2D(filters=3, kernel_size=2, padding='same')
    inputs = Input(shape=input_shape[1:])
    x = reused_layer(inputs)
    x = reused_layer(x)
    x = ReLU()(x)
    return keras.Model(inputs=inputs, outputs=x)


def get_multiple_outputs_model(input_shape):
    inputs = Input(shape=input_shape[1:])
    x = Conv2D(filters=2, kernel_size=3)(inputs)
    x = BatchNormalization()(x)
    out1 = ReLU(max_value=6.0)(x)
    x2 = Conv2D(2, 4)(out1)
    out2 = ReLU(max_value=6.0)(x2)
    return keras.Model(inputs=inputs, outputs=[out1, out2])


def get_multiple_outputs_to_intermediate_node_model(input_shape):
    inputs = Input(shape=input_shape[1:])
    x = Conv2D(filters=2, kernel_size=3)(inputs)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6.0)(x)
    x_split = tf.split(x, num_or_size_splits=2, axis=-1)
    outputs = x_split[0] + x_split[1]
    return keras.Model(inputs=inputs, outputs=outputs)


def get_multiple_inputs_model(input_shape):
    inputs = Input(shape=input_shape[1:])
    inputs2 = Input(shape=input_shape[1:])

    x = Conv2D(filters=2, kernel_size=3)(inputs)
    x2 = Conv2D(filters=2, kernel_size=3)(inputs2)

    outputs = x + x2
    return keras.Model(inputs=[inputs, inputs2], outputs=outputs)


def representative_dataset(input_shape, num_of_inputs=1):
    yield [np.random.randn(*input_shape).astype(np.float32)] * num_of_inputs


def get_expected_shape(t_shape, granularity, node_type):
    if granularity == hessian_common.HessianInfoGranularity.PER_ELEMENT:
        return (1, *t_shape)
    elif granularity == hessian_common.HessianInfoGranularity.PER_TENSOR:
        return (1, 1)
    else:
        return (1, t_shape[-1] * t_shape[2]) if node_type == DepthwiseConv2D else \
            (1, t_shape[2]) if node_type == Conv2DTranspose else \
                (1, t_shape[-1])


class TestHessianInfoCalculatorBase(unittest.TestCase):

    def _setup(self, layer, input_shape=(1, 8, 8, 3)):
        in_model = basic_model(input_shape, layer=layer)
        keras_impl = KerasImplementation()
        _repr_dataset = functools.partial(representative_dataset,
                                          input_shape=input_shape)
        graph = prepare_graph_with_configs(in_model,
                                           keras_impl,
                                           DEFAULT_KERAS_INFO,
                                           _repr_dataset,
                                           generate_keras_tpc)

        return graph, _repr_dataset, keras_impl


class TestHessianInfoCalculatorWeights(TestHessianInfoCalculatorBase):

    def _test_hessian_scores(self, hessian_info, target_nodes, granularity, num_scores=1):
        request = hessian_common.TraceHessianRequest(mode=hessian_common.HessianMode.WEIGHTS,
                                                     granularity=granularity,
                                                     target_nodes=target_nodes)

        info = hessian_info.fetch_hessian(request, num_scores)

        scores = []
        for i, target_node in enumerate(target_nodes):
            # The call for fetch_hessian returns the requested number of scores for each target node.
            # Since in this test we request computation for a single node, we need to extract its results from the list.
            self.assertTrue(isinstance(info, list))

            node_info = info[i]

            self.assertTrue(len(node_info) == num_scores, f"fetched {num_scores} score but {len(node_info)} scores were fetched")
            score = np.mean(np.stack(node_info), axis=0)

            kernel_attr_name = [w for w in target_node.weights if KERNEL in w]
            self.assertTrue(len(kernel_attr_name) == 1, "Expecting exactly 1 kernel attribute.")
            expected_shape = (
                get_expected_shape(target_node.weights[kernel_attr_name[0]].shape, granularity, target_node.type))

            self.assertTrue(isinstance(score, np.ndarray), f"scores expected to be a numpy array but is {type(score)}")
            self.assertTrue(score.shape == expected_shape,
                            f"Tensor shape is expected to be {expected_shape} but has shape {score.shape}")  # per tensor

            scores.append(score)

        return scores

    def test_conv2d_granularity(self):
        graph, _repr_dataset, keras_impl = self._setup(layer=Conv2D(filters=2, kernel_size=3))
        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        interest_points = [n for n in sorted_graph_nodes if len(n.weights) > 0]
        hessian_service = hessian_common.HessianInfoService(graph=graph, representative_dataset_gen=_repr_dataset,
                                                            fw_impl=keras_impl)
        self._test_hessian_scores(hessian_service,
                                  interest_points,
                                  granularity=hessian_common.HessianInfoGranularity.PER_TENSOR)
        self._test_hessian_scores(hessian_service,
                                  interest_points,
                                  granularity=hessian_common.HessianInfoGranularity.PER_OUTPUT_CHANNEL)
        self._test_hessian_scores(hessian_service,
                                  interest_points,
                                  granularity=hessian_common.HessianInfoGranularity.PER_ELEMENT)
        del hessian_service

    def test_dense_granularity(self):
        graph, _repr_dataset, keras_impl = self._setup(layer=Dense(2), input_shape=(1, 8))
        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        interest_points = [n for n in sorted_graph_nodes if n.type == Dense]
        hessian_service = hessian_common.HessianInfoService(graph=graph, representative_dataset_gen=_repr_dataset,
                                                            fw_impl=keras_impl)

        self._test_hessian_scores(hessian_service,
                                  interest_points,
                                  granularity=hessian_common.HessianInfoGranularity.PER_TENSOR)
        self._test_hessian_scores(hessian_service,
                                  interest_points,
                                  granularity=hessian_common.HessianInfoGranularity.PER_OUTPUT_CHANNEL)
        self._test_hessian_scores(hessian_service,
                                  interest_points,
                                  granularity=hessian_common.HessianInfoGranularity.PER_ELEMENT)
        del hessian_service

    def test_conv2dtranspose_granularity(self):
        graph, _repr_dataset, keras_impl = self._setup(layer=Conv2DTranspose(filters=2, kernel_size=3))
        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        interest_points = [n for n in sorted_graph_nodes if len(n.weights) > 0]
        hessian_service = hessian_common.HessianInfoService(graph=graph, representative_dataset_gen=_repr_dataset,
                                                            fw_impl=keras_impl)

        self._test_hessian_scores(hessian_service,
                                  interest_points,
                                  granularity=hessian_common.HessianInfoGranularity.PER_TENSOR)
        self._test_hessian_scores(hessian_service,
                                  interest_points,
                                  granularity=hessian_common.HessianInfoGranularity.PER_OUTPUT_CHANNEL)
        self._test_hessian_scores(hessian_service,
                                  interest_points,
                                  granularity=hessian_common.HessianInfoGranularity.PER_ELEMENT)
        del hessian_service

    def test_depthwiseconv2d_granularity(self):
        graph, _repr_dataset, keras_impl = self._setup(layer=DepthwiseConv2D(kernel_size=3))
        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        interest_points = [n for n in sorted_graph_nodes if len(n.weights) > 0]
        hessian_service = hessian_common.HessianInfoService(graph=graph, representative_dataset_gen=_repr_dataset,
                                                            fw_impl=keras_impl)

        self._test_hessian_scores(hessian_service,
                                  interest_points,
                                  granularity=hessian_common.HessianInfoGranularity.PER_TENSOR)
        self._test_hessian_scores(hessian_service,
                                  interest_points,
                                  granularity=hessian_common.HessianInfoGranularity.PER_OUTPUT_CHANNEL)
        self._test_hessian_scores(hessian_service,
                                  interest_points,
                                  granularity=hessian_common.HessianInfoGranularity.PER_ELEMENT)
        del hessian_service

    def test_reused_layer(self):
        input_shape = (1, 8, 8, 3)
        in_model = reused_model(input_shape)
        _repr_dataset = functools.partial(representative_dataset,
                                          input_shape=input_shape)

        keras_impl = KerasImplementation()
        graph = prepare_graph_with_configs(in_model,
                                           keras_impl,
                                           DEFAULT_KERAS_INFO,
                                           _repr_dataset,
                                           generate_keras_tpc)

        sorted_graph_nodes = graph.get_topo_sorted_nodes()

        # Two nodes representing the same reused layer
        interest_points = [n for n in sorted_graph_nodes if n.is_match_type(Conv2D)]
        self.assertTrue(len(interest_points) == 2, f"Expected to find 2 Conv2D nodes but found {len(interest_points)}")

        hessian_service = hessian_common.HessianInfoService(graph=graph, representative_dataset_gen=_repr_dataset,
                                                            fw_impl=keras_impl)
        node1_approx = self._test_hessian_scores(hessian_service,
                                                 [interest_points[0]],
                                                 granularity=hessian_common.HessianInfoGranularity.PER_TENSOR)[0]
        node2_approx = self._test_hessian_scores(hessian_service,
                                                 [interest_points[1]],
                                                 granularity=hessian_common.HessianInfoGranularity.PER_TENSOR)[0]
        self.assertTrue(np.all(node1_approx == node2_approx), f'Approximations of nodes of a reused layer '
                                                              f'should be equal')

        # Expecting call for count_saved_info_of_request for reused node to be with a reconstructed request with the
        # node's representative group member for its reuse group
        with self.assertRaises(Exception) as e:
            node1_count = hessian_service.count_saved_info_of_request(
                hessian_common.TraceHessianRequest(target_nodes=[interest_points[0]],
                                                   mode=hessian_common.HessianMode.WEIGHTS,
                                                   granularity=hessian_common.HessianInfoGranularity.PER_TENSOR))
        self.assertTrue("Expecting the Hessian request to include only non-reused nodes at this point"
                        in str(e.exception))

        node2_count = hessian_service.count_saved_info_of_request(
            hessian_common.TraceHessianRequest(target_nodes=[interest_points[1]],
                                               mode=hessian_common.HessianMode.WEIGHTS,
                                               granularity=hessian_common.HessianInfoGranularity.PER_TENSOR))
        self.assertTrue(node2_count[interest_points[1]] == 1)
        self.assertTrue(len(hessian_service.trace_hessian_request_to_score_list) == 1)
        del hessian_service

    #########################################################
    # The following part checks different possible graph
    # properties (#inputs/#outputs, for example).
    ########################################################

    def _test_advanced_graph(self, float_model, _repr_dataset):
        ########################################################################
        # Since we want to test some models with different properties (e.g., multiple inputs/outputs)
        # we can no longer assume we're fetching interest point #1 like in the linear ops
        # tests. Instead, this function assumes the first Conv2D interest point is the interest point.
        #######################################################################
        keras_impl = KerasImplementation()
        graph = prepare_graph_with_configs(float_model,
                                           keras_impl,
                                           DEFAULT_KERAS_INFO,
                                           _repr_dataset,
                                           generate_keras_tpc)

        sorted_graph_nodes = graph.get_topo_sorted_nodes()

        # This test assumes the first Conv2D interest point is the node that
        # we fetch its scores and test their shapes correctness.
        interest_points = [n for n in sorted_graph_nodes if n.type == Conv2D]
        hessian_service = hessian_common.HessianInfoService(graph=graph, representative_dataset_gen=_repr_dataset,
                                                            fw_impl=keras_impl)
        self._test_hessian_scores(hessian_service,
                                  interest_points,
                                  granularity=hessian_common.HessianInfoGranularity.PER_TENSOR)
        self._test_hessian_scores(hessian_service,
                                  interest_points,
                                  granularity=hessian_common.HessianInfoGranularity.PER_OUTPUT_CHANNEL)
        self._test_hessian_scores(hessian_service,
                                  interest_points,
                                  granularity=hessian_common.HessianInfoGranularity.PER_ELEMENT)

        del hessian_service

    def test_multiple_inputs(self):
        input_shape = (1, 8, 8, 3)
        in_model = get_multiple_inputs_model(input_shape)
        _repr_dataset = functools.partial(representative_dataset,
                                          input_shape=input_shape,
                                          num_of_inputs=2)
        self._test_advanced_graph(in_model, _repr_dataset)

    def test_multiple_outputs(self):
        input_shape = (1, 8, 8, 3)
        in_model = get_multiple_outputs_model(input_shape)
        _repr_dataset = functools.partial(representative_dataset,
                                          input_shape=input_shape)
        self._test_advanced_graph(in_model, _repr_dataset)

    def test_multiple_outputs_to_intermediate_node(self):
        input_shape = (1, 8, 8, 3)
        in_model = get_multiple_outputs_to_intermediate_node_model(input_shape)
        _repr_dataset = functools.partial(representative_dataset,
                                          input_shape=input_shape)
        self._test_advanced_graph(in_model, _repr_dataset)


class TestHessianInfoCalculatorActivation(TestHessianInfoCalculatorBase):

    def _test_hessian_scores(self, hessian_info, target_nodes, granularity, num_scores=1):
        request = hessian_common.TraceHessianRequest(mode=hessian_common.HessianMode.ACTIVATION,
                                                     granularity=granularity,
                                                     target_nodes=target_nodes)
        info = hessian_info.fetch_hessian(request, num_scores)

        # The call for fetch_hessian returns the requested number of scores for each target node.
        self.assertTrue(isinstance(info, list))

        # currently, activation support only per-tensor Hessian
        expected_shape = (1, 1)
        scores = []

        for i, target_node in enumerate(target_nodes):
            node_info = info[i]
            self.assertTrue(len(node_info) == num_scores,
                            f"Requested {num_scores} score but {len(node_info)} scores were fetched")

            score = np.mean(np.stack(node_info), axis=0)

            self.assertTrue(score.shape == expected_shape,
                            f"Tensor shape is expected to be {expected_shape} but has shape {score.shape}")  # per tensor

            scores.append(score)

        return scores

    def test_conv2d_granularity(self):
        graph, _repr_dataset, keras_impl = self._setup(layer=Conv2D(filters=2, kernel_size=3))
        sorted_graph_nodes = graph.get_topo_sorted_nodes()

        interest_points = [n for n in sorted_graph_nodes if len(n.weights) > 0]

        hessian_service = hessian_common.HessianInfoService(graph=graph, representative_dataset_gen=_repr_dataset,
                                                            fw_impl=keras_impl)
        self._test_hessian_scores(hessian_service,
                                  interest_points,
                                  granularity=hessian_common.HessianInfoGranularity.PER_TENSOR)

        del hessian_service

    def test_dense_granularity(self):
        graph, _repr_dataset, keras_impl = self._setup(layer=Dense(2), input_shape=(1, 8))
        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        interest_points = [n for n in sorted_graph_nodes if len(n.weights) > 0]
        hessian_service = hessian_common.HessianInfoService(graph=graph, representative_dataset_gen=_repr_dataset,
                                                            fw_impl=keras_impl)

        self._test_hessian_scores(hessian_service,
                                  interest_points,
                                  granularity=hessian_common.HessianInfoGranularity.PER_TENSOR)

        del hessian_service

    def test_conv2dtranspose_granularity(self):
        graph, _repr_dataset, keras_impl = self._setup(layer=Conv2DTranspose(filters=2, kernel_size=3))
        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        interest_points = [n for n in sorted_graph_nodes if len(n.weights) > 0]
        hessian_service = hessian_common.HessianInfoService(graph=graph, representative_dataset_gen=_repr_dataset,
                                                            fw_impl=keras_impl)

        self._test_hessian_scores(hessian_service,
                                  interest_points,
                                  granularity=hessian_common.HessianInfoGranularity.PER_TENSOR)

        del hessian_service

    def test_depthwiseconv2d_granularity(self):
        graph, _repr_dataset, keras_impl = self._setup(layer=DepthwiseConv2D(kernel_size=3))
        sorted_graph_nodes = graph.get_topo_sorted_nodes()
        interest_points = [n for n in sorted_graph_nodes if len(n.weights) > 0]
        hessian_service = hessian_common.HessianInfoService(graph=graph, representative_dataset_gen=_repr_dataset,
                                                            fw_impl=keras_impl)

        self._test_hessian_scores(hessian_service,
                                  interest_points,
                                  granularity=hessian_common.HessianInfoGranularity.PER_TENSOR)

        del hessian_service

    def test_reused_layer(self):
        input_shape = (1, 8, 8, 3)
        in_model = reused_model(input_shape)
        _repr_dataset = functools.partial(representative_dataset,
                                          input_shape=input_shape)

        keras_impl = KerasImplementation()
        graph = prepare_graph_with_configs(in_model,
                                           keras_impl,
                                           DEFAULT_KERAS_INFO,
                                           _repr_dataset,
                                           generate_keras_tpc)

        sorted_graph_nodes = graph.get_topo_sorted_nodes()

        # Two nodes representing the same reused layer
        interest_points = [n for n in sorted_graph_nodes if n.is_match_type(Conv2D)]
        self.assertTrue(len(interest_points) == 2, f"Expected to find 2 Conv2D nodes but found {len(interest_points)}")

        hessian_service = hessian_common.HessianInfoService(graph=graph, representative_dataset_gen=_repr_dataset,
                                                            fw_impl=keras_impl)

        node1_approx = self._test_hessian_scores(hessian_service,
                                                 [interest_points[0]],
                                                 granularity=hessian_common.HessianInfoGranularity.PER_TENSOR)[0]
        node2_approx = self._test_hessian_scores(hessian_service,
                                                 [interest_points[1]],
                                                 granularity=hessian_common.HessianInfoGranularity.PER_TENSOR)[0]

        self.assertTrue(np.all(node1_approx == node2_approx), f'Approximations of nodes of a reused layer '
                                                              f'should be equal')

        # Expecting call for count_saved_info_of_request for reused node to be with a reconstructed request with the
        # node's representative group member for its reuse group
        with self.assertRaises(Exception) as e:
            node1_count = hessian_service.count_saved_info_of_request(
                hessian_common.TraceHessianRequest(target_nodes=[interest_points[0]],
                                                   mode=hessian_common.HessianMode.ACTIVATION,
                                                   granularity=hessian_common.HessianInfoGranularity.PER_TENSOR))
        self.assertTrue("Expecting the Hessian request to include only non-reused nodes at this point"
                        in str(e.exception))

        node2_count = hessian_service.count_saved_info_of_request(
            hessian_common.TraceHessianRequest(target_nodes=[interest_points[1]],
                                               mode=hessian_common.HessianMode.ACTIVATION,
                                               granularity=hessian_common.HessianInfoGranularity.PER_TENSOR))
        self.assertTrue(node2_count[interest_points[1]] == 1)

        self.assertTrue(len(hessian_service.trace_hessian_request_to_score_list) == 1)

        del hessian_service

    #########################################################
    # The following part checks different possible graph
    # properties (#inputs/#outputs, for example).
    ########################################################

    def _test_advanced_graph(self, float_model, _repr_dataset):
        ########################################################################
        # Since we want to test some models with different properties (e.g., multiple inputs/outputs)
        # we can no longer assume we're fetching interest point #1 like in the linear ops
        # tests. Instead, this function assumes the first Conv2D interest point is the interest point.
        #######################################################################
        keras_impl = KerasImplementation()
        graph = prepare_graph_with_configs(float_model,
                                           keras_impl,
                                           DEFAULT_KERAS_INFO,
                                           _repr_dataset,
                                           generate_keras_tpc)

        sorted_graph_nodes = graph.get_topo_sorted_nodes()

        # This test assumes the first Conv2D interest point is the node that
        # we fetch its scores and test their shapes correctness.
        interest_points = [n for n in sorted_graph_nodes if n.type == Conv2D]
        hessian_service = hessian_common.HessianInfoService(graph=graph, representative_dataset_gen=_repr_dataset,
                                                            fw_impl=keras_impl)
        self._test_hessian_scores(hessian_service,
                                  interest_points,
                                  granularity=hessian_common.HessianInfoGranularity.PER_TENSOR)

        del hessian_service

    def test_multiple_inputs(self):
        input_shape = (1, 8, 8, 3)
        in_model = get_multiple_inputs_model(input_shape)
        _repr_dataset = functools.partial(representative_dataset,
                                          input_shape=input_shape,
                                          num_of_inputs=2)
        self._test_advanced_graph(in_model, _repr_dataset)

    def test_multiple_outputs(self):
        input_shape = (1, 8, 8, 3)
        in_model = get_multiple_outputs_model(input_shape)
        _repr_dataset = functools.partial(representative_dataset,
                                          input_shape=input_shape)
        self._test_advanced_graph(in_model, _repr_dataset)

    def test_multiple_outputs_to_intermediate_node(self):
        input_shape = (1, 8, 8, 3)
        in_model = get_multiple_outputs_to_intermediate_node_model(input_shape)
        _repr_dataset = functools.partial(representative_dataset,
                                          input_shape=input_shape)
        self._test_advanced_graph(in_model, _repr_dataset)

    def test_activation_hessian_output_exception(self):
        graph, _repr_dataset, keras_impl = self._setup(layer=Conv2D(filters=2, kernel_size=3))
        hessian_service = hessian_common.HessianInfoService(graph=graph, representative_dataset_gen=_repr_dataset,
                                                            fw_impl=keras_impl)
        with self.assertRaises(Exception) as e:
            request = hessian_common.TraceHessianRequest(granularity=hessian_common.HessianInfoGranularity.PER_TENSOR,
                                                         mode=hessian_common.HessianMode.ACTIVATION,
                                                         target_nodes=[graph.get_outputs()[0].node])
            _ = hessian_service.fetch_hessian(request, required_size=1)

        self.assertTrue("Trying to compute activation Hessian approximation with respect to the model output"
                        in str(e.exception))

        del hessian_service
