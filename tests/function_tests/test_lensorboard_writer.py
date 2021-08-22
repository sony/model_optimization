# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


import unittest

import network_optimization_package as snop
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
        snop.common.Logger.set_log_file('/tmp/')
        model = MobileNet()
        snop.keras_post_training_quantization(model, random_datagen, n_iter=1)

    def test_tensorboard_log_dir(self):
        self.assertTrue(os.path.exists(os.path.join(snop.common.Logger.LOG_PATH, 'tensorboard_logs')))

    def test_tensorboard_initial_graph_num_of_nodes(self):
        events_dir = os.path.join(snop.common.Logger.LOG_PATH, 'tensorboard_logs/')
        events_files = glob.glob(events_dir+'initial_graph/*events*')
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
