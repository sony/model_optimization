# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig, \
    NodeActivationQuantizationConfig
from model_compression_toolkit import qunatizers_infrastructure as qi, QuantizationConfig
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs


class BasePytorchInfrastructureTest:

    def __init__(self,
                 unit_test,
                 num_calibration_iter=1,
                 val_batch_size=1,
                 num_of_inputs=1,
                 input_shape=(3, 8, 8)):
        self.unit_test = unit_test
        self.val_batch_size = val_batch_size
        self.num_calibration_iter = num_calibration_iter
        self.num_of_inputs = num_of_inputs
        self.input_shape = (val_batch_size,) + input_shape

    def generate_inputs(self):
        return [np.random.randn(*in_shape) for in_shape in self.get_input_shapes()]

    def get_input_shapes(self):
        return [self.input_shape for _ in range(self.num_of_inputs)]

    def get_dispatcher(self, weight_quantizers=None, activation_quantizers=None):
        return qi.PytorchNodeQuantizationDispatcher(weight_quantizers, activation_quantizers)

    def get_wrapper(self, layer, dispatcher):
        return qi.PytorchQuantizationWrapper(layer, dispatcher)
