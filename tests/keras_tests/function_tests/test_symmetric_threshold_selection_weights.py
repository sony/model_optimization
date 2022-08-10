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
import unittest

import numpy as np
from keras import Input, Model
from keras.layers import Conv2D, Conv2DTranspose

import model_compression_toolkit as mct
from model_compression_toolkit import QuantizationConfig, QuantizationErrorMethod, CoreConfig
from model_compression_toolkit.core.common.constants import THRESHOLD
from model_compression_toolkit.core.common.model_collector import ModelCollector
from model_compression_toolkit.core.common.quantization.quantization_analyzer import analyzer_graph
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_computation import \
    calculate_quantization_params
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_keras_tpc
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model


def get_uniform_weights(kernel, in_channels, out_channels):
    return np.array([i - np.round((in_channels * kernel * kernel * out_channels) / 2) for i in
                     range(in_channels * kernel * kernel * out_channels)]).reshape(
        [out_channels, kernel, kernel, in_channels]).transpose(1, 2, 3, 0)


def create_network():
    num_conv_channels = 4
    kernel = 3
    conv_w = get_uniform_weights(kernel, num_conv_channels, num_conv_channels)

    inputs = Input(shape=(16, 16, num_conv_channels))
    x = Conv2D(num_conv_channels, kernel, use_bias=False)(inputs)
    outputs = Conv2DTranspose(num_conv_channels, kernel, use_bias=False)(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.layers[1].set_weights([conv_w])
    model.layers[2].set_weights([conv_w])
    return model


class TestSymmetricThresholdSelectionWeights(unittest.TestCase):

    def test_per_channel_weights_symmetric_threshold_selection_no_clipping(self):
        self.run_test_for_threshold_method(QuantizationErrorMethod.NOCLIPPING)

    def test_weights_symmetric_threshold_selection_no_clipping(self):
        self.run_test_for_threshold_method(QuantizationErrorMethod.NOCLIPPING, per_channel=False)

    def test_per_channel_weights_symmetric_threshold_selection_mse(self):
        self.run_test_for_threshold_method(QuantizationErrorMethod.MSE)

    def test_weights_symmetric_threshold_selection_mse(self):
        self.run_test_for_threshold_method(QuantizationErrorMethod.MSE, per_channel=False)

    def test_per_channel_weights_symmetric_threshold_selection_mae(self):
        self.run_test_for_threshold_method(QuantizationErrorMethod.MAE)

    def test_weights_symmetric_threshold_selection_mae(self):
        self.run_test_for_threshold_method(QuantizationErrorMethod.MAE, per_channel=False)

    def test_per_channel_weights_symmetric_threshold_selection_lp(self):
        self.run_test_for_threshold_method(QuantizationErrorMethod.LP)

    def test_weights_symmetric_threshold_selection_lp(self):
        self.run_test_for_threshold_method(QuantizationErrorMethod.LP, per_channel=False)

    def test_per_channel_weights_symmetric_threshold_selection_kl(self):
        self.run_test_for_threshold_method(QuantizationErrorMethod.KL)

    def test_weights_symmetric_threshold_selection_kl(self):
        self.run_test_for_threshold_method(QuantizationErrorMethod.KL, per_channel=False)

    def run_test_for_threshold_method(self, threshold_method, per_channel=True):
        qc = QuantizationConfig(weights_error_method=threshold_method,
                                weights_per_channel_threshold=per_channel)
        core_config = CoreConfig(n_iter=1, quantization_config=qc)

        tp = generate_test_tp_model({
            'weights_quantization_method': mct.target_platform.QuantizationMethod.SYMMETRIC})
        tpc = generate_keras_tpc(name="symmetric_threshold_selection_test", tp_model=tp)

        fw_info = DEFAULT_KERAS_INFO
        in_model = create_network()
        keras_impl = KerasImplementation()
        graph = keras_impl.model_reader(in_model, None)  # model reading
        graph.set_fw_info(fw_info)
        graph.set_tpc(tpc)
        graph = set_quantization_configuration_to_graph(graph=graph,
                                                        quant_config=core_config.quantization_config,
                                                        mixed_precision_enable=core_config.mixed_precision_enable)
        for node in graph.nodes:
            node.prior_info = keras_impl.get_node_prior_info(node=node,
                                                             fw_info=fw_info,
                                                             graph=graph)
        analyzer_graph(keras_impl.attach_sc_to_node,
                       graph,
                       fw_info)

        mi = ModelCollector(graph, fw_info=DEFAULT_KERAS_INFO, fw_impl=keras_impl)
        for i in range(10):
            mi.infer([np.random.randn(1, 16, 16, 4)])

        calculate_quantization_params(graph,
                                      fw_info,
                                      fw_impl=keras_impl)

        nodes_list = list(graph.nodes)
        conv1_threshold = nodes_list[0].candidates_quantization_cfg[0].weights_quantization_cfg.weights_quantization_params[THRESHOLD]
        conv2_threshold = nodes_list[1].candidates_quantization_cfg[0].weights_quantization_cfg.weights_quantization_params[THRESHOLD]
        conv1_threshold_log = np.log2(conv1_threshold)
        conv2_threshold_log = np.log2(conv2_threshold)
        self.assertFalse(np.array_equal(conv1_threshold_log, conv1_threshold_log.astype(int)),
                         msg=f"First conv layer threshold {conv1_threshold} is a power of 2")
        self.assertFalse(np.array_equal(conv2_threshold_log, conv2_threshold_log.astype(int)),
                         msg=f"Second conv layer threshold {conv2_threshold} is a power of 2")


if __name__ == '__main__':
    unittest.main()
