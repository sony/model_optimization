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
from unittest.mock import Mock
from typing import Callable

from model_compression_toolkit.core import DebugConfig, CoreConfig


class BaseMCTBypassTest(abc.ABC):

    @abc.abstractmethod
    def _build_test_model(self):
        """ build framework model for test"""
        raise NotImplementedError()

    @abc.abstractmethod
    def _assert_models_equal(self, model, out_model):
        """ assert that two models are identical (have the exact same sub modules and parameters"""
        raise NotImplementedError()

    def _test_mct_bypass(self, api_func: Callable):
        model = self._build_test_model()
        core_config = CoreConfig(debug_config=DebugConfig(bypass=True))
        out_model, user_info = api_func(model, Mock(), Mock(), core_config=core_config)

        self._assert_models_equal(model, out_model)
