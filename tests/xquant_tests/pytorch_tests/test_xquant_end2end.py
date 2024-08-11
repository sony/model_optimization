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
import sys

import subprocess

import glob

import os

import unittest
from functools import partial
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.compat.proto.graph_pb2 import GraphDef

import model_compression_toolkit as mct
from mct_quantizers import PytorchQuantizationWrapper
from model_compression_toolkit.constants import PYTORCH
from model_compression_toolkit.target_platform_capabilities.constants import IMX500_TP_MODEL
from model_compression_toolkit.xquant.common.similarity_functions import DEFAULT_SIMILARITY_METRICS_NAMES
from model_compression_toolkit.xquant.common.xquant_config import XQuantConfig
from model_compression_toolkit.xquant.pytorch.facade_xquant_report import xquant_report_pytorch_experimental
from model_compression_toolkit.xquant.common.constants import OUTPUT_SIMILARITY_METRICS_REPR, \
    OUTPUT_SIMILARITY_METRICS_VAL, INTERMEDIATE_SIMILARITY_METRICS_REPR, INTERMEDIATE_SIMILARITY_METRICS_VAL, \
    XQUANT_REPR, XQUANT_VAL, CUT_MEMORY_ELEMENTS, CUT_TOTAL_SIZE

def random_data_gen(shape=(3, 8, 8), use_labels=False, num_inputs=1, batch_size=2, num_iter=2):
    if use_labels:
        for _ in range(num_iter):
            yield [[torch.randn(batch_size, *shape)] * num_inputs, torch.randn(batch_size)]
    else:
        for _ in range(num_iter):
            yield [torch.randn(batch_size, *shape)] * num_inputs

class BaseTestEnd2EndPytorchXQuant(unittest.TestCase):

    def setUp(self):
        self.float_model = self.get_model_to_test()
        self.repr_dataset = partial(random_data_gen, shape=self.get_input_shape())
        self.quantized_model, _ = mct.ptq.pytorch_post_training_quantization(in_module=self.float_model,
                                                                             representative_data_gen=self.repr_dataset,
                                                                             target_platform_capabilities=self.get_tpc(),
                                                                             core_config=self.get_core_config())

        self.validation_dataset = partial(random_data_gen, use_labels=True)
        self.tmpdir = tempfile.mkdtemp()
        self.xquant_config = XQuantConfig(report_dir=self.tmpdir)

    def get_input_shape(self):
        return (3, 8, 8)

    def get_core_config(self):
        return mct.core.CoreConfig(debug_config=mct.core.DebugConfig(simulate_scheduler=True))

    def get_tpc(self):
        return mct.get_target_platform_capabilities(PYTORCH, IMX500_TP_MODEL, "v2")

    def get_model_to_test(self):
        class BaseModelTest(torch.nn.Module):
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
        return BaseModelTest()

    def test_xquant_report_output_metrics(self):
        self.xquant_config.custom_similarity_metrics = None
        result = xquant_report_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        self.assertIn(OUTPUT_SIMILARITY_METRICS_REPR, result)
        self.assertEqual(len(result[OUTPUT_SIMILARITY_METRICS_REPR]), len(DEFAULT_SIMILARITY_METRICS_NAMES))
        self.assertIn(OUTPUT_SIMILARITY_METRICS_VAL, result)
        self.assertEqual(len(result[OUTPUT_SIMILARITY_METRICS_VAL]), len(DEFAULT_SIMILARITY_METRICS_NAMES))

    def test_intermediate_metrics(self):
        self.xquant_config.custom_similarity_metrics = None
        result = xquant_report_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )

        self.assertIn(INTERMEDIATE_SIMILARITY_METRICS_REPR, result)
        linear_layers = [n for n,m in self.quantized_model.named_modules() if isinstance(m, PytorchQuantizationWrapper)]

        self.assertIn(linear_layers[0], result[INTERMEDIATE_SIMILARITY_METRICS_REPR])
        for k,v in result[INTERMEDIATE_SIMILARITY_METRICS_REPR].items():
            self.assertEqual(len(v), len(DEFAULT_SIMILARITY_METRICS_NAMES))

        self.assertIn(INTERMEDIATE_SIMILARITY_METRICS_VAL, result)
        for k,v in result[INTERMEDIATE_SIMILARITY_METRICS_VAL].items():
            self.assertEqual(len(v), len(DEFAULT_SIMILARITY_METRICS_NAMES))

    def test_custom_metric(self):
        self.xquant_config.custom_similarity_metrics = {'mae': lambda x,y: torch.nn.L1Loss()(x,y).item()}
        result = xquant_report_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )

        self.assertIn(OUTPUT_SIMILARITY_METRICS_REPR, result)
        self.assertEqual(len(result[OUTPUT_SIMILARITY_METRICS_REPR]), len(DEFAULT_SIMILARITY_METRICS_NAMES) + 1)
        self.assertIn("mae", result[OUTPUT_SIMILARITY_METRICS_REPR])

        self.assertIn(INTERMEDIATE_SIMILARITY_METRICS_REPR, result)
        for k,v in result[INTERMEDIATE_SIMILARITY_METRICS_REPR].items():
            self.assertEqual(len(v), len(DEFAULT_SIMILARITY_METRICS_NAMES) + 1)
            self.assertIn("mae", v)

    def test_tensorboard_graph(self):
        self.xquant_config.custom_similarity_metrics = None
        result = xquant_report_pytorch_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        events_dir = os.path.join(self.xquant_config.report_dir, 'xquant')
        initial_graph_events_files = glob.glob(events_dir + '/*events*')
        initial_graph_event = initial_graph_events_files[0]
        efl = event_file_loader.LegacyEventFileLoader(initial_graph_event).Load()
        for e in efl:
            if len(e.graph_def) > 0:  # skip events with no graph_def such as event version
                g = GraphDef().FromString(e.graph_def)
        for node in g.node:
            if node.device == 'PytorchQuantizationWrapper':
                self.assertIn(XQUANT_REPR, str(node))
                self.assertIn(XQUANT_VAL, str(node))
                self.assertIn(CUT_MEMORY_ELEMENTS, str(node))
                self.assertIn(CUT_TOTAL_SIZE, str(node))


# Test with Conv2D without BatchNormalization and without Activation
class TestXQuantReportModel2(BaseTestEnd2EndPytorchXQuant):

    def get_model_to_test(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

            def forward(self, x):
                x1 = self.conv1(x)
                x = x + x1
                x = F.softmax(x, dim=1)
                return x

        return Model()


# Test with Multiple Convolution Layers and an Addition Operator
class TestXQuantReportModel3(BaseTestEnd2EndPytorchXQuant):
    def get_model_to_test(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                x = x1 + x2
                x = self.conv3(x)
                x = F.softmax(x, dim=1)
                return x

        return Model()



class TestXQuantReportModelMBv2(BaseTestEnd2EndPytorchXQuant):

    def get_input_shape(self):
        return (3, 224, 224)

    def get_model_to_test(self):
        from torchvision.models.mobilenetv2 import MobileNetV2
        return MobileNetV2()


class TestXQuantReportModelMBv3(BaseTestEnd2EndPytorchXQuant):

    def get_input_shape(self):
        return (3, 224, 224)

    def get_model_to_test(self):
        from torchvision.models.mobilenet import mobilenet_v3_small
        return mobilenet_v3_small()


if __name__ == '__main__':
    unittest.main()
