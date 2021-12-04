# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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

import model_compression_toolkit as mct
import numpy as np
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorboard.compat.proto.graph_pb2 import GraphDef
import os
import glob
from tensorboard.backend.event_processing import event_file_loader


def random_datagen():
    return [np.random.random((1, 224, 224, 3))]


class TestLogger(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mct.common.Logger.set_log_file('/tmp/')
        model = MobileNet()
        mct.keras_post_training_quantization(model, random_datagen, n_iter=1)

    def test_tensorboard_log_dir(self):
        self.assertTrue(os.path.exists(os.path.join(mct.common.Logger.LOG_PATH, 'tensorboard_logs')))

    def test_tensorboard_initial_graph_num_of_nodes(self):
        events_dir = os.path.join(mct.common.Logger.LOG_PATH, 'tensorboard_logs/')
        events_files = glob.glob(events_dir + 'initial_graph/*events*')
        self.assertTrue(len(events_files) == 1)  # Make sure there is only event file in 'initial_graph' subdir

        event_filepath = events_files[0]
        efl = event_file_loader.LegacyEventFileLoader(event_filepath).Load()
        for e in efl:
            if len(e.graph_def) > 0:  # skip events with no graph_def such as event version
                g = GraphDef().FromString(e.graph_def)
        nodes_in_model = len(MobileNet().layers)
        nodes_in_graph = len(g.node)
        self.assertTrue(nodes_in_graph == nodes_in_model)


if __name__ == '__main__':
    unittest.main()
