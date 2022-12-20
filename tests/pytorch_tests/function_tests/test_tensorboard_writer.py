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
from torchvision.models import mobilenet_v2

import model_compression_toolkit as mct
from model_compression_toolkit.core import common
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_pytorch_tpc
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs
from tests.common_tests.helpers.generate_test_tp_model import generate_tp_model_with_activation_mp
from tests.pytorch_tests.model_tests.feature_models.mixed_precision_activation_test import MixedPrecisionNet
from tests.pytorch_tests.model_tests.feature_models.multiple_outputs_node_test import MultipleOutputsNet


def random_datagen():
    return [np.random.random((1, 3, 224, 224))]


def multiple_random_datagen():
    return [np.random.random((1, 3, 224, 224)), np.random.random((1, 3, 224, 224))]


class BasePytorchTestLogger(unittest.TestCase):

    """
    This is the base test of PyTorch Logger.
    """

    @classmethod
    def setUpClass(cls):
        mct.core.common.Logger.set_log_file('/tmp/')

    def test_tensorboard_log_dir(self):
        self.assertTrue(os.path.exists(os.path.join(mct.core.common.Logger.LOG_PATH, 'tensorboard_logs')))

    def test_tensorboard_initial_graph(self):
        events_dir = os.path.join(mct.core.common.Logger.LOG_PATH, 'tensorboard_logs/')
        events_files = glob.glob(events_dir + 'initial_graph/*events*')
        self.assertTrue(len(events_files) == 1)  # Make sure there is only event file in 'initial_graph' subdir


class PytorchTestLogger(BasePytorchTestLogger):

    def setUp(self):
        self.model = MixedPrecisionNet([(1, 3, 224, 224)])
        mct.pytorch_post_training_quantization(self.model, random_datagen, n_iter=1, analyze_similarity=True)


class PytorchMultipleOutputsTestLogger(BasePytorchTestLogger):

    def setUp(self):
        self.model = MultipleOutputsNet()
        mct.pytorch_post_training_quantization(self.model, multiple_random_datagen, n_iter=1, analyze_similarity=True)


class PytorchMixedPrecisionTestLogger(BasePytorchTestLogger):

    def setUp(self):
        self.model = MixedPrecisionNet([(1, 3, 224, 224)])
        kpi = mct.KPI()
        base_config, _ = get_op_quantization_configs()
        tpc_model = generate_tp_model_with_activation_mp(
                base_cfg=base_config,
                mp_bitwidth_candidates_list=[(8, 8), (8, 4), (8, 2),
                                             (4, 8), (4, 4), (4, 2),
                                             (2, 8), (2, 4), (2, 2)])
        tpc = generate_pytorch_tpc(name='mp_pytorch_tpc', tp_model=tpc_model)
        mct.pytorch_post_training_quantization_mixed_precision(self.model, random_datagen,
                                                               target_platform_capabilities=tpc,
                                                               target_kpi=kpi,
                                                               n_iter=1, analyze_similarity=True)


if __name__ == '__main__':
    unittest.main()
