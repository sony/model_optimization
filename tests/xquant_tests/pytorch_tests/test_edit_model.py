#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
#

import unittest
from functools import partial
import tempfile
import torch

import model_compression_toolkit as mct
from xquant import XQuantConfig

from xquant import xquant_report_pytorch_experimental

class ModelToTest(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.identity = torch.nn.Identity()

    def forward(self, x):
        x = self.identity(x)
        return x


def random_data_gen(shape=(2, 3, 8, 8), use_labels=False):
    if use_labels:
        for _ in range(2):
            yield [[torch.randn(*shape)], torch.randn(shape[0])]
    else:
        for _ in range(2):
            yield [torch.randn(*shape)]


class TestXQuantReport(unittest.TestCase):

    def setUp(self):
        self.float_model = ModelToTest()
        self.core_config = mct.core.CoreConfig()
        self.repr_dataset = random_data_gen
        self.quantized_model, _ = mct.ptq.pytorch_post_training_quantization(in_module=self.float_model,
                                                                             core_config=self.core_config,
                                                                             representative_data_gen=self.repr_dataset)

        self.validation_dataset = partial(random_data_gen, use_labels=True)
        self.tmpdir = tempfile.mkdtemp()
        self.xquant_config = XQuantConfig(report_dir=self.tmpdir)

    def test_disable_quantization(self):
        disable_act_action = mct.core.network_editor.ChangeFinalActivationQuantConfigAttr(enable_activation_quantization=False)
        self.xquant_config.edit_rules = [
            mct.core.network_editor.EditRule(
                filter=mct.core.network_editor.NodeNameFilter('x'),
                action=disable_act_action),
            ]

        result = xquant_report_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.core_config,
            self.xquant_config
        )

        self.assertEqual(result['output_metrics_repr']['mse'], 0)





if __name__ == '__main__':
    unittest.main()
