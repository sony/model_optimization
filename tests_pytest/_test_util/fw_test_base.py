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
from typing import Callable

from model_compression_toolkit.core import FrameworkInfo, QuantizationConfig, CoreConfig
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner


class BaseFWIntegrationTest(abc.ABC):
    """ Base class providing utils for integration / e2e tests. """

    fw_info: FrameworkInfo
    fw_impl: FrameworkImplementation
    attach_to_fw_func: Callable

    def run_graph_preparation(self, model, datagen, tpc, quant_config=None,
                              mp: bool = False, gptq: bool = False, bit_width_config=None):
        quant_config = quant_config or QuantizationConfig()
        graph = graph_preparation_runner(model,
                                         datagen,
                                         quantization_config=quant_config,
                                         fw_info=self.fw_info,
                                         fw_impl=self.fw_impl,
                                         fqc=self.attach_to_fw_func(tpc),
                                         mixed_precision_enable=mp,
                                         running_gptq=gptq,
                                         bit_width_config=bit_width_config)

        return graph


