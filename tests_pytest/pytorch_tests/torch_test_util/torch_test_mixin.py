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
from typing import List, Tuple

import torch

from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
    AttachTpcToPytorch
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation


class TorchFwMixin:
    """ Mixin helper class containing keras-specific definitions. """
    fw_info = DEFAULT_PYTORCH_INFO
    fw_impl = PytorchImplementation()
    attach_to_fw_func = AttachTpcToPytorch().attach

    def get_basic_data_gen(self, shapes: List[Tuple]):
        """ Generate a basic data generator. """
        def f():
            yield [torch.randn(shape, dtype=torch.float32) for shape in shapes]
        return f
