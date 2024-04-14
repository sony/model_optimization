# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
import tensorflow as tf
from tensorflow.keras import layers

import model_compression_toolkit as mct
from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.constants import THRESHOLD, RANGE_MAX
from model_compression_toolkit.core.common.hessian import HessianInfoService, TraceHessianRequest, HessianMode, \
    HessianInfoGranularity
from model_compression_toolkit.core.common.model_collector import ModelCollector
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_computation import \
    calculate_quantization_params
from model_compression_toolkit.core.keras.constants import KERNEL
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_configs


def model_gen():
    inputs = layers.Input(shape=[8, 8, 3])
    x = layers.Conv2D(2, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(8)(x)
    return tf.keras.models.Model(inputs=inputs, outputs=x)


def representative_dataset():
    yield [np.random.randn(1, 8, 8, 3).astype(np.float32)]


def get_tpc(quant_method, per_channel):
    tp = generate_test_tp_model(edit_params_dict={
        'weights_quantization_method': quant_method,
        'weights_per_channel_threshold': per_channel})
    tpc = generate_keras_tpc(name="hmse_params_selection_test", tp_model=tp)

    return tpc


class TestParamSelectionWithHMSE(unittest.TestCase):
    def _setup_with_args(self, quant_method, per_channel):
        self.qc = QuantizationConfig(weights_error_method=mct.core.QuantizationErrorMethod.HMSE)
        self.float_model = model_gen()
        self.keras_impl = KerasImplementation()
        self.fw_info = DEFAULT_KERAS_INFO
        self.quant_method = quant_method
        self.per_channel = per_channel

        self.graph = prepare_graph_with_configs(self.float_model, self.keras_impl, self.fw_info,
                                                representative_dataset,
                                                lambda name, _tp: get_tpc(quant_method, per_channel),
                                                qc=self.qc,
                                                running_gptq=True  # to enable HMSE in params calculation
                                                )

        self.his = HessianInfoService(graph=self.graph,
                                      representative_dataset=representative_dataset,
                                      fw_impl=self.keras_impl)

        mi = ModelCollector(self.graph,
                            fw_impl=self.keras_impl,
                            fw_info=self.fw_info,
                            qc=self.qc)

        for i in range(10):
            mi.infer([np.random.randn(1, 8, 8, 3)])

    def _verify_params_calculation_execution(self, param_name):
        def _run_node_verification(node_type):
            node = [n for n in self.graph.nodes if n.type == node_type]
            self.assertTrue(len(node) == 1, f"Expecting exactly 1 {node_type} node in test model.")
            node = node[0]

            kernel_attr_qparams = (
                node.candidates_quantization_cfg[0].weights_quantization_cfg.get_attr_config(KERNEL))
            self.assertTrue(kernel_attr_qparams.weights_quantization_params.get(param_name) is not None,
                            f"Expecting {node_type} node parameters {param_name} to be initialized.")

            expected_hessian_request = TraceHessianRequest(mode=HessianMode.WEIGHTS,
                                                           granularity=HessianInfoGranularity.PER_ELEMENT,
                                                           target_node=node)

            self.assertTrue(self.his.count_saved_info_of_request(expected_hessian_request) > 0,
                            f"No Hessian-based scores were computed for node {node}, "
                            "but expected parameters selection to run with HMSE.")

        _run_node_verification(layers.Conv2D)
        _run_node_verification(layers.Dense)

    def test_pot_threshold_selection_hmse_per_channel(self):

        self._setup_with_args(quant_method=mct.target_platform.QuantizationMethod.POWER_OF_TWO, per_channel=True)
        calculate_quantization_params(self.graph, hessian_info_service=self.his, num_hessian_samples=1)
        self._verify_params_calculation_execution(THRESHOLD)

    def test_pot_threshold_selection_hmse_per_tensor(self):

        self._setup_with_args(quant_method=mct.target_platform.QuantizationMethod.POWER_OF_TWO, per_channel=False)
        calculate_quantization_params(self.graph, hessian_info_service=self.his, num_hessian_samples=1)
        self._verify_params_calculation_execution(THRESHOLD)

    def test_symmetric_threshold_selection_hmse_per_channel(self):

        self._setup_with_args(quant_method=mct.target_platform.QuantizationMethod.SYMMETRIC, per_channel=True)
        calculate_quantization_params(self.graph, hessian_info_service=self.his, num_hessian_samples=1)
        self._verify_params_calculation_execution(THRESHOLD)

    def test_symmetric_threshold_selection_hmse_per_tensor(self):

        self._setup_with_args(quant_method=mct.target_platform.QuantizationMethod.SYMMETRIC, per_channel=False)
        calculate_quantization_params(self.graph, hessian_info_service=self.his, num_hessian_samples=1)
        self._verify_params_calculation_execution(THRESHOLD)

    def test_usniform_threshold_selection_hmse_per_channel(self):

        self._setup_with_args(quant_method=mct.target_platform.QuantizationMethod.UNIFORM, per_channel=True)
        calculate_quantization_params(self.graph, hessian_info_service=self.his, num_hessian_samples=1)
        self._verify_params_calculation_execution(RANGE_MAX)

    def test_uniform_threshold_selection_hmse_per_tensor(self):

        self._setup_with_args(quant_method=mct.target_platform.QuantizationMethod.UNIFORM, per_channel=False)
        calculate_quantization_params(self.graph, hessian_info_service=self.his, num_hessian_samples=1)
        self._verify_params_calculation_execution(RANGE_MAX)


if __name__ == '__main__':
    unittest.main()
