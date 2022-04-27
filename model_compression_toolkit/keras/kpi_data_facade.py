# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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

from typing import Callable

from model_compression_toolkit.common.hardware_representation import FrameworkHardwareModel
from model_compression_toolkit.common.kpi_data import compute_kpi_data
from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.common.quantization.quantization_config import DEFAULTCONFIG
from model_compression_toolkit.keras.quantization_facade import KERAS_DEFAULT_MODEL

import importlib


if importlib.util.find_spec("tensorflow") is not None\
        and importlib.util.find_spec("tensorflow_model_optimization") is not None:
    from model_compression_toolkit.keras.default_framework_info import DEFAULT_KERAS_INFO
    from model_compression_toolkit.keras.keras_implementation import KerasImplementation
    from tensorflow.keras.models import Model



    def keras_kpi_data(in_model: Model,
                       representative_data_gen: Callable,
                       quant_config: QuantizationConfig = DEFAULTCONFIG,
                       fw_info: FrameworkInfo = DEFAULT_KERAS_INFO,
                       fw_hw_model: FrameworkHardwareModel = KERAS_DEFAULT_MODEL):

        fw_impl = KerasImplementation()

        return compute_kpi_data(in_model,
                                representative_data_gen,
                                quant_config,
                                fw_hw_model,
                                fw_info,
                                fw_impl)
