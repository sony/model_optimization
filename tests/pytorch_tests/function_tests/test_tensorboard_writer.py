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

import model_compression_toolkit as mct
import numpy as np
from torchvision.models import mobilenet_v2
import os
import glob

from model_compression_toolkit.logger import Logger


def random_datagen():
    yield [np.random.random((1, 3, 224, 224))]


class PytorchTestLogger(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        Logger.set_log_file('/tmp/')
        model = mobilenet_v2(pretrained=True)
        core_config = mct.CoreConfig(debug_config=mct.DebugConfig(analyze_similarity=True))
        mct.ptq.pytorch_post_training_quantization_experimental(model, random_datagen, core_config=core_config)

    @classmethod
    def tearDownClass(cls) -> None:
        mct.core.Logger.shutdown()

    def test_tensorboard_log_dir(self):
        self.assertTrue(os.path.exists(os.path.join(mct.core.Logger.LOG_PATH, 'tensorboard_logs')))

    def test_tensorboard_initial_graph(self):
        events_dir = os.path.join(mct.core.Logger.LOG_PATH, 'tensorboard_logs/')
        events_files = glob.glob(events_dir + 'initial_graph/*events*')
        self.assertTrue(len(events_files) == 1)  # Make sure there is only event file in 'initial_graph' subdir


if __name__ == '__main__':
    unittest.main()
