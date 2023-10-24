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
import copy

import glob
import os
import unittest

import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.compat.proto.graph_pb2 import GraphDef

import model_compression_toolkit as mct
from model_compression_toolkit.constants import TENSORFLOW
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    DEFAULT_MIXEDPRECISION_CONFIG
from model_compression_toolkit.core.common.visualization.final_config_visualizer import \
    ActivationFinalBitwidthConfigVisualizer
from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs
from tests.common_tests.helpers.generate_test_tp_model import generate_tp_model_with_activation_mp
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_set_bit_widths

keras = tf.keras
layers = keras.layers


def random_datagen():
    return [np.random.random((1, 8, 8, 3))]


def SingleOutputNet():
    inputs = layers.Input(shape=(8, 8, 3))
    x = layers.Dense(2)(inputs)
    x = layers.Conv2D(2, 4)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    output = layers.Dense(2)(x)
    return keras.Model(inputs=inputs, outputs=output)


def MultipleOutputsNet():
    inputs = layers.Input(shape=(8, 8, 3))
    x = layers.Dense(2)(inputs)
    x = layers.Conv2D(2, 4)(x)
    x = layers.BatchNormalization()(x)
    out1 = layers.ReLU(max_value=6.0)(x)
    out2 = layers.Dense(2)(out1)
    return keras.Model(inputs=inputs, outputs=[out1, out2])


class TestFileLogger(unittest.TestCase):
    """
    This is the test of Keras Logger.
    This test checks logging into file.
    """

    def setUp(self):
        Logger.set_log_file('/tmp/')

    def tensorboard_log_dir(self):
        self.assertTrue(os.path.exists(os.path.join(Logger.LOG_PATH, 'tensorboard_logs')))

    def tensorboard_initial_graph_num_of_nodes(self, num_event_files, event_to_test):
        events_dir = os.path.join(Logger.LOG_PATH, 'tensorboard_logs/')

        initial_graph_events_files = glob.glob(events_dir + 'initial_graph/*events*')
        self.assertTrue(len(initial_graph_events_files) == num_event_files)  # Make sure there is only 2 event files in
        # 'initial_graph' subdir

        initial_graph_event = initial_graph_events_files[event_to_test]

        efl = event_file_loader.LegacyEventFileLoader(initial_graph_event).Load()
        for e in efl:
            if len(e.graph_def) > 0:  # skip events with no graph_def such as event version
                g = GraphDef().FromString(e.graph_def)
        nodes_in_model = len(self.model.layers)
        nodes_in_graph = len(g.node)
        self.assertTrue(nodes_in_graph == nodes_in_model)

    def plot_tensor_sizes(self):
        model = SingleOutputNet()
        base_config, _ = get_op_quantization_configs()
        tpc_model = generate_tp_model_with_activation_mp(
            base_cfg=base_config,
            mp_bitwidth_candidates_list=[(8, 8), (8, 4), (8, 2),
                                         (4, 8), (4, 4), (4, 2),
                                         (2, 8), (2, 4), (2, 2)])
        tpc = generate_keras_tpc(name='mp_keras_tpc', tp_model=tpc_model)

        # Hessian service assumes core should be initialized. This test does not do it, so we disable the use of hessians in MP
        cfg = copy.deepcopy(DEFAULT_MIXEDPRECISION_CONFIG)
        cfg.use_grad_based_weights=False

        # compare max tensor size with plotted max tensor size
        tg = prepare_graph_set_bit_widths(in_model=model,
                                          fw_impl=KerasImplementation(),
                                          fw_info=DEFAULT_KERAS_INFO,
                                          representative_data_gen=random_datagen,
                                          tpc=tpc,
                                          network_editor=[],
                                          quant_config=cfg,
                                          target_kpi=mct.core.KPI(),
                                          n_iter=1, analyze_similarity=True)
        tensors_sizes = [4.0 * n.get_total_output_params() / 1000000.0
                         for n in tg.get_sorted_activation_configurable_nodes()]  # in MB
        max_tensor_size = max(tensors_sizes)

        # plot tensor sizes
        activation_conf_nodes_bitwidth = tg.get_final_activation_config()
        visual = ActivationFinalBitwidthConfigVisualizer(activation_conf_nodes_bitwidth)
        fig = visual.plot_tensor_sizes(tg)
        figure_max_tensor_size = max([rect._height for rect in fig.axes[0].get_children()[:len(
            activation_conf_nodes_bitwidth)]])
        self.assertTrue(figure_max_tensor_size == max_tensor_size)

    def test_steps_by_order(self):
        # Test Single Output Mixed Precision model Logger
        self.model = SingleOutputNet()
        tpc = mct.get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)

        def rep_data():
            yield [np.random.randn(1, 8, 8, 3)]

        mp_qc = mct.core.MixedPrecisionQuantizationConfigV2(num_of_images=1,
                                                            use_grad_based_weights=False)
        core_config = mct.core.CoreConfig(mixed_precision_config=mp_qc)
        quantized_model, _ = mct.ptq.keras_post_training_quantization_experimental(self.model,
                                                                               rep_data,
                                                                               target_kpi=mct.core.KPI(np.inf),
                                                                               core_config=core_config,
                                                                               target_platform_capabilities=tpc,
                                                                               new_experimental_exporter=True)

        self.tensorboard_initial_graph_num_of_nodes(num_event_files=1, event_to_test=0)

        # Test Logger file created
        self.tensorboard_log_dir()

        # Test Multiple Outputs model Logger
        self.model = MultipleOutputsNet()
        quantized_model, _ = mct.ptq.keras_post_training_quantization_experimental(self.model,
                                                                               rep_data,
                                                                               target_kpi=mct.core.KPI(np.inf),
                                                                               core_config=core_config,
                                                                               target_platform_capabilities=tpc,
                                                                               new_experimental_exporter=True)

        # Test tensor size plotting
        self.plot_tensor_sizes()

        # Disable Logger
        Logger.LOG_PATH = None


if __name__ == '__main__':
    unittest.main()
