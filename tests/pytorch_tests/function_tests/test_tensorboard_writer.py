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
import glob
import os
import unittest

import numpy as np
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.compat.proto.graph_pb2 import GraphDef

import model_compression_toolkit as mct
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    DEFAULT_MIXEDPRECISION_CONFIG
from model_compression_toolkit.core.common.visualization.final_config_visualizer import \
    ActivationFinalBitwidthConfigVisualizer
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_pytorch_tpc
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs
from tests.common_tests.helpers.generate_test_tp_model import generate_tp_model_with_activation_mp
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_set_bit_widths
from tests.pytorch_tests.model_tests.feature_models.mixed_precision_activation_test import MixedPrecisionNet


def random_datagen():
    return [np.random.random((1, 3, 224, 224))]


class BasePytorchTestLogger(unittest.TestCase):
    """
    This is the base test of PyTorch Logger.
    """

    @classmethod
    def setUpClass(cls):
        common.Logger.set_log_file('/tmp/')
        model = MixedPrecisionNet([(1, 3, 224, 224)])
        mct.pytorch_post_training_quantization(model, random_datagen, n_iter=1, analyze_similarity=True)
        # cls.addClassCleanup(common.Logger.set_log_file, None)

    def test_tensorboard_log_dir(self):
        self.assertTrue(os.path.exists(os.path.join(common.Logger.LOG_PATH, 'tensorboard_logs')))
        common.Logger.set_log_file(None)

    def test_tensorboard_initial_graph(self):
        events_dir = os.path.join(common.Logger.LOG_PATH, 'tensorboard_logs/')
        events_files = glob.glob(events_dir + 'initial_graph/*events*')
        self.assertTrue(len(events_files) == 1)  # Make sure there is only event file in 'initial_graph' subdir

        event_filepath = events_files[0]
        efl = event_file_loader.LegacyEventFileLoader(event_filepath).Load()
        for e in efl:
            if len(e.graph_def) > 0:  # skip events with no graph_def such as event version
                g = GraphDef().FromString(e.graph_def)
        nodes_in_initial_graph = len(g.node)

        # check nodes in graph after bn folding = original -1
        events_files = glob.glob(events_dir + 'pre_statistics_collection_substitutions/*events*')
        self.assertTrue(len(events_files) == 1)  # Make sure there is only event file in
        # 'pre_statistics_collection_substitutions' subdir

        event_filepath = events_files[0]
        efl = event_file_loader.LegacyEventFileLoader(event_filepath).Load()
        for e in efl:
            if len(e.graph_def) > 0:  # skip events with no graph_def such as event version
                g = GraphDef().FromString(e.graph_def)
        nodes_in_bn_folding_graph = len(g.node)
        # Graph after BN folding
        self.assertTrue(nodes_in_bn_folding_graph == nodes_in_initial_graph - 1)


class PytorchTestLogger(BasePytorchTestLogger):

    def setUp(self):
        self.model = MixedPrecisionNet([(1, 3, 224, 224)])
        mct.pytorch_post_training_quantization(self.model, random_datagen, n_iter=1, analyze_similarity=True)

#
# class PytorchMixedPrecisionTensorSizesTestLogger(unittest.TestCase):
#     def setUp(self):
#         self.model = MixedPrecisionNet([(1, 3, 224, 224)])
#         base_config, _ = get_op_quantization_configs()
#         tpc_model = generate_tp_model_with_activation_mp(
#             base_cfg=base_config,
#             mp_bitwidth_candidates_list=[(8, 8), (8, 4), (8, 2),
#                                          (4, 8), (4, 4), (4, 2),
#                                          (2, 8), (2, 4), (2, 2)])
#         self.tpc = generate_pytorch_tpc(name='mp_pytorch_tpc', tp_model=tpc_model)
#
#     def test_plot_tensor_sizes(self):
#         # compare max tensor size with plotted max tensor size
#         tg = prepare_graph_set_bit_widths(in_model=self.model,
#                                           fw_impl=PytorchImplementation(),
#                                           fw_info=DEFAULT_PYTORCH_INFO,
#                                           representative_data_gen=random_datagen,
#                                           tpc=self.tpc,
#                                           network_editor=[],
#                                           quant_config=DEFAULT_MIXEDPRECISION_CONFIG,
#                                           target_kpi=mct.KPI(),
#                                           n_iter=1, analyze_similarity=True)
#         tensors_sizes = [4.0 * n.get_total_output_params() / 1000000.0
#                          for n in tg.get_sorted_activation_configurable_nodes()]  # in MB
#         max_tensor_size = max(tensors_sizes)
#
#         # plot tensor sizes
#         activation_conf_nodes_bitwidth = tg.get_final_activation_config()
#         visual = ActivationFinalBitwidthConfigVisualizer(activation_conf_nodes_bitwidth)
#         fig = visual.plot_tensor_sizes(tg)
#         figure_max_tensor_size = max([rect._height for rect in fig.axes[0].get_children()[:len(
#             activation_conf_nodes_bitwidth)]])
#         self.assertTrue(figure_max_tensor_size == max_tensor_size)
#


if __name__ == '__main__':
    unittest.main()
