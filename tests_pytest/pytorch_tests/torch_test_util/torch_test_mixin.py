# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
import abc
from typing import List, Tuple

import torch
from mct_quantizers import PytorchActivationQuantizationHolder, PytorchQuantizationWrapper

from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
    AttachTpcToPytorch
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from tests_pytest._test_util.fw_test_base import BaseFWIntegrationTest


class TorchFwMixin:
    """ Mixin helper class containing torch-specific definitions.
        This is handy when the test has a fw-agnostic base class, for example:

        BaseFooTester(BaseFWIntegrationTest):
          ...
        TorchFooTester(BaseFooTester, TorchFwMixin):
          ...
    """
    fw_info = DEFAULT_PYTORCH_INFO
    fw_impl = PytorchImplementation()
    attach_to_fw_func = AttachTpcToPytorch().attach

    @staticmethod
    def get_basic_data_gen(shapes: List[Tuple]):
        """ Generate a basic data generator. """
        def f():
            yield [torch.randn(shape, dtype=torch.float32) for shape in shapes]
        return f

    @staticmethod
    def fetch_activation_holder_quantizer(model, layer_name):
        layer = getattr(model, model.node_to_activation_quantization_holder[layer_name])
        assert isinstance(layer, PytorchActivationQuantizationHolder)
        return layer.activation_holder_quantizer

    @staticmethod
    def fetch_weight_quantizer(layer, weight_name):
        assert isinstance(layer, PytorchQuantizationWrapper)
        return layer.weights_quantizers[weight_name]

    @staticmethod
    def fetch_model_layers_by_cls(model, cls):
        """ Fetch layers from torch module by layer class type.  """
        return [m for m in model.modules() if isinstance(m, cls)]


class BaseTorchIntegrationTest(BaseFWIntegrationTest, TorchFwMixin, abc.ABC):
    """ Base class for Torch integration tests. """
    pass
