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
from mct_quantizers import PytorchQuantizationWrapper
from xquant.common.framework_report_utils import DEFAULT_METRICS_NAMES
from xquant.common.xquant_config import XQuantConfig
from xquant.pytorch.facade_xquant_report import xquant_report_pytorch_experimental
from xquant.common.constants import OUTPUT_METRICS_REPR, OUTPUT_METRICS_VAL, INTERMEDIATE_METRICS_REPR, INTERMEDIATE_METRICS_VAL



class ModelToTest(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        self.bn = torch.nn.BatchNorm2d(num_features=3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
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
        self.repr_dataset = random_data_gen
        self.quantized_model, _ = mct.ptq.pytorch_post_training_quantization(in_module=self.float_model,
                                                                             representative_data_gen=self.repr_dataset)

        self.validation_dataset = partial(random_data_gen, use_labels=True)
        self.tmpdir = tempfile.mkdtemp()
        self.xquant_config = XQuantConfig(report_dir=self.tmpdir)

    def test_xquant_report_output_metrics_repr(self):
        self.xquant_config.custom_similarity_metrics = None
        result = xquant_report_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )

        self.assertIn(OUTPUT_METRICS_REPR, result)
        self.assertEqual(len(result[OUTPUT_METRICS_REPR]), len(DEFAULT_METRICS_NAMES))  # Assuming no custom metrics collected


    def test_xquant_report_output_metrics_val(self):
        self.xquant_config.custom_similarity_metrics = None
        result = xquant_report_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )

        self.assertIn(OUTPUT_METRICS_VAL, result)
        self.assertEqual(len(result[OUTPUT_METRICS_VAL]), len(DEFAULT_METRICS_NAMES))  # Assuming no custom metrics collected



    def test_custom_output_metric(self):
        self.xquant_config.custom_similarity_metrics = {'mae': lambda x,y: torch.nn.L1Loss()(x,y).item()}
        result = xquant_report_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )

        self.assertIn(OUTPUT_METRICS_REPR, result)
        self.assertEqual(len(result[OUTPUT_METRICS_REPR]), len(DEFAULT_METRICS_NAMES) + 1)
        self.assertIn("mae", result[OUTPUT_METRICS_REPR])

    def test_intermediate_metrics_repr(self):
        self.xquant_config.custom_similarity_metrics = None
        result = xquant_report_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )

        self.assertIn(INTERMEDIATE_METRICS_REPR, result)
        linear_layers = [n for n,m in self.quantized_model.named_modules() if isinstance(m, PytorchQuantizationWrapper)]
        self.assertEqual(len(linear_layers), 1, msg=f"Expected to find one linear layer. Found {len(linear_layers)}")
        self.assertIn(linear_layers[0], result[INTERMEDIATE_METRICS_REPR])

        for k,v in result[INTERMEDIATE_METRICS_REPR].items():
            self.assertEqual(len(v), len(DEFAULT_METRICS_NAMES))  # Assuming no custom metrics collected


    def test_intermediate_metrics_val(self):
        self.xquant_config.custom_similarity_metrics = None
        result = xquant_report_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        self.assertIn(INTERMEDIATE_METRICS_VAL, result)
        for k,v in result[INTERMEDIATE_METRICS_VAL].items():
            self.assertEqual(len(v), len(DEFAULT_METRICS_NAMES))  # Assuming no custom metrics collected

    def test_custom_intermediate_metrics(self):
        self.xquant_config.custom_similarity_metrics = {'mae': lambda x,y: torch.nn.L1Loss()(x,y).item()}
        result = xquant_report_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )

        self.assertIn(INTERMEDIATE_METRICS_REPR, result)
        for k,v in result[INTERMEDIATE_METRICS_REPR].items():
            self.assertEqual(len(v), len(DEFAULT_METRICS_NAMES)+1)
            self.assertIn("mae", v)


if __name__ == '__main__':
    unittest.main()
