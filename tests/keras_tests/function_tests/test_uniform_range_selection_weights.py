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
import numpy as np
import unittest

from keras import Input, Model
from keras.layers import Conv2D, Conv2DTranspose

import model_compression_toolkit as mct
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core import QuantizationConfig, QuantizationErrorMethod
from model_compression_toolkit.constants import RANGE_MIN, RANGE_MAX
from model_compression_toolkit.core.keras.constants import KERNEL
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2keras import \
    AttachTpcToKeras
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from tests.common_tests.helpers.generate_test_tpc import generate_test_tpc
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_quantization_parameters


def get_random_weights(kernel, in_channels, out_channels):
    return np.random.normal(size=[kernel, kernel, in_channels, out_channels])


def create_network():
    num_conv_channels = 4
    kernel = 3
    conv_w1 = get_random_weights(kernel, num_conv_channels, num_conv_channels)
    conv_w2 = get_random_weights(kernel, num_conv_channels, num_conv_channels)

    inputs = Input(shape=(16, 16, num_conv_channels))
    x = Conv2D(num_conv_channels, kernel, use_bias=False)(inputs)
    outputs = Conv2DTranspose(num_conv_channels, kernel, use_bias=False)(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.layers[1].set_weights([conv_w1])
    model.layers[2].set_weights([conv_w2])
    return model


def representative_dataset():
    yield [np.random.randn(1, 16, 16, 4).astype(np.float32)]


def get_tpc(per_channel):
    tp = generate_test_tpc({
        'weights_quantization_method': QuantizationMethod.UNIFORM,
        'weights_per_channel_threshold': per_channel})
    tpc = generate_keras_tpc(name="uniform_range_selection_test", tpc=tp)

    return tpc


class TestUniformRangeSelectionWeights(unittest.TestCase):

    def test_per_channel_weights_uniform_range_selection_no_clipping(self):
        self.run_test_for_threshold_method(QuantizationErrorMethod.NOCLIPPING)

    def test_weights_uniform_range_selection_no_clipping(self):
        self.run_test_for_threshold_method(QuantizationErrorMethod.NOCLIPPING, per_channel=False)

    def test_per_channel_weights_uniform_range_selection_mse(self):
        self.run_test_for_threshold_method(QuantizationErrorMethod.MSE)

    def test_weights_uniform_range_selection_mse(self):
        self.run_test_for_threshold_method(QuantizationErrorMethod.MSE, per_channel=False)

    def test_per_channel_weights_uniform_range_selection_mae(self):
        self.run_test_for_threshold_method(QuantizationErrorMethod.MAE)

    def test_weights_uniform_range_selection_mae(self):
        self.run_test_for_threshold_method(QuantizationErrorMethod.MAE, per_channel=False)

    def test_per_channel_weights_uniform_range_selection_lp(self):
        self.run_test_for_threshold_method(QuantizationErrorMethod.LP)

    def test_weights_uniform_range_selection_lp(self):
        self.run_test_for_threshold_method(QuantizationErrorMethod.LP, per_channel=False)

    def test_per_channel_weights_uniform_range_selection_kl(self):
        self.run_test_for_threshold_method(QuantizationErrorMethod.KL)

    def test_weights_uniform_range_selection_kl(self):
        self.run_test_for_threshold_method(QuantizationErrorMethod.KL, per_channel=False)

    def run_test_for_threshold_method(self, threshold_method, per_channel=True):
        qc = QuantizationConfig(weights_error_method=threshold_method)

        in_model = create_network()
        graph = prepare_graph_with_quantization_parameters(in_model, KerasImplementation(),
                                                           representative_dataset,
                                                           lambda name, _tp: get_tpc(per_channel),
                                                           qc=qc, input_shape=(1, 16, 16, 4),
                                                           attach2fw=AttachTpcToKeras(), )

        nodes_list = list(graph.nodes)
        conv1_min = nodes_list[0].candidates_quantization_cfg[0].weights_quantization_cfg.get_attr_config(KERNEL).weights_quantization_params[RANGE_MIN].flatten()
        conv2_min = nodes_list[1].candidates_quantization_cfg[0].weights_quantization_cfg.get_attr_config(KERNEL).weights_quantization_params[RANGE_MIN].flatten()
        conv1_max = nodes_list[0].candidates_quantization_cfg[0].weights_quantization_cfg.get_attr_config(KERNEL).weights_quantization_params[RANGE_MAX].flatten()
        conv2_max = nodes_list[1].candidates_quantization_cfg[0].weights_quantization_cfg.get_attr_config(KERNEL).weights_quantization_params[RANGE_MAX].flatten()

        for range_min, range_max in list(zip(conv1_min, conv1_max)):
            self.assertTrue(range_min <= 0 <= range_max,
                            msg=f"First conv layer quantization range ({range_min}, {range_max}) does not include 0")
        for range_min, range_max in list(zip(conv2_min, conv2_max)):
            self.assertTrue(range_min <= 0 <= range_max,
                            msg=f"First conv layer quantization range ({range_min}, {range_max}) does not include 0")


if __name__ == '__main__':
    unittest.main()
