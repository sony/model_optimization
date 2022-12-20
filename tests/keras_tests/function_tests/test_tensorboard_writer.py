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


import glob
import os
import unittest

import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.compat.proto.graph_pb2 import GraphDef

import model_compression_toolkit as mct
from model_compression_toolkit.core import common
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_keras_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_tp_model_with_activation_mp

keras = tf.keras
layers = keras.layers


def random_datagen():
    return [np.random.random((1, 224, 224, 3))]


def SingleOutputNet():
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Dense(20)(inputs)
    x = layers.ReLU(max_value=6.0)(x)
    output = layers.Dense(20)(x)
    return keras.Model(inputs=inputs, outputs=output)


def MultipleOutputsNet():
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Dense(20)(inputs)
    x = layers.ReLU(max_value=6.0)(x)
    outputs = layers.Dense(20)(x)
    outputs = tf.split(outputs, num_or_size_splits=20, axis=-1)
    return keras.Model(inputs=inputs, outputs=[outputs[0], outputs[4], outputs[2]])


class BaseTestLogger(unittest.TestCase):

    """
    This is the base test of Keras Logger.
    """

    @classmethod
    def setUpClass(cls):
        common.Logger.set_log_file('/tmp/')

    def test_tensorboard_log_dir(self):
        self.assertTrue(os.path.exists(os.path.join(common.Logger.LOG_PATH, 'tensorboard_logs')))

    def test_tensorboard_initial_graph_num_of_nodes(self):
        events_dir = os.path.join(common.Logger.LOG_PATH, 'tensorboard_logs/')
        events_files = glob.glob(events_dir + 'initial_graph/*events*')
        self.assertTrue(len(events_files) == 1)  # Make sure there is only event file in 'initial_graph' subdir

        event_filepath = events_files[0]
        efl = event_file_loader.LegacyEventFileLoader(event_filepath).Load()
        for e in efl:
            if len(e.graph_def) > 0:  # skip events with no graph_def such as event version
                g = GraphDef().FromString(e.graph_def)
        nodes_in_model = len(self.model.layers)
        nodes_in_graph = len(g.node)
        self.assertTrue(nodes_in_graph == nodes_in_model)


class TestLogger(BaseTestLogger):

    def setUp(self):
        self.model = SingleOutputNet()
        mct.keras_post_training_quantization(self.model, random_datagen, n_iter=1, analyze_similarity=True)


class MultipleOutputsTestLogger(BaseTestLogger):

    def setUp(self):
        self.model = MultipleOutputsNet()
        mct.keras_post_training_quantization(self.model, random_datagen, n_iter=1, analyze_similarity=True)


class MixedPrecisionTestLogger(BaseTestLogger):

    def setUp(self):
        self.model = SingleOutputNet()
        kpi = mct.KPI()
        base_config, _ = get_op_quantization_configs()
        tpc_model = generate_tp_model_with_activation_mp(
                base_cfg=base_config,
                mp_bitwidth_candidates_list=[(8, 8), (8, 4), (8, 2),
                                             (4, 8), (4, 4), (4, 2),
                                             (2, 8), (2, 4), (2, 2)])
        tpc = generate_keras_tpc(name='mp_keras_tpc', tp_model=tpc_model)
        mct.keras_post_training_quantization_mixed_precision(self.model, random_datagen,
                                                             target_kpi=kpi,
                                                             n_iter=1,
                                                             target_platform_capabilities=tpc,
                                                             analyze_similarity=True)


if __name__ == '__main__':
    unittest.main()
