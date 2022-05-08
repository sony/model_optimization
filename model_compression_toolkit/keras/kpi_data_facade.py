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

from model_compression_toolkit import KPI, MixedPrecisionQuantizationConfig
from model_compression_toolkit.common import Logger
from model_compression_toolkit.common.constants import TENSORFLOW
from model_compression_toolkit.common.target_platform import TargetPlatformCapabilities
from model_compression_toolkit.common.mixed_precision.kpi_data import compute_kpi_data
from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    DEFAULT_MIXEDPRECISION_CONFIG
from model_compression_toolkit.keras.constants import DEFAULT_TP_MODEL

import importlib


if importlib.util.find_spec("tensorflow") is not None\
        and importlib.util.find_spec("tensorflow_model_optimization") is not None:
    from model_compression_toolkit.keras.default_framework_info import DEFAULT_KERAS_INFO
    from model_compression_toolkit.keras.keras_implementation import KerasImplementation
    from tensorflow.keras.models import Model

    from model_compression_toolkit import get_target_platform_capabilities

    KERAS_DEFAULT_TPC = get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)


    def keras_kpi_data(in_model: Model,
                       representative_data_gen: Callable,
                       quant_config: MixedPrecisionQuantizationConfig = DEFAULT_MIXEDPRECISION_CONFIG,
                       fw_info: FrameworkInfo = DEFAULT_KERAS_INFO,
                       target_platform_capabilities: TargetPlatformCapabilities = KERAS_DEFAULT_TPC) -> KPI:
        """
        Computes KPI data that can be used to calculate the desired target KPI for mixed-precision quantization.
        Builds the computation graph from the given model and hw modeling, and uses it to compute the KPI data.

        Args:
            in_model (Model): Keras model to quantize.
            representative_data_gen (Callable): Dataset used for calibration.
            quant_config (MixedPrecisionQuantizationConfig): MixedPrecisionQuantizationConfig containing parameters
            of how the model should be quantized.
            fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g.,
            kernel channels indices, groups of layers by how they should be quantized, etc.). `Default Keras info
            <https://github.com/sony/model_optimization/blob/21e21c95ca25a31874a5be7af9dd2dd5da8f3a10
            /model_compression_toolkit/keras/default_framework_info.py#L113>`_
            target_platform_capabilities (TargetPlatformCapabilities): TargetPlatformCapabilities to optimize the
            Keras model according to. `Default Keras info
            <https://github.com/sony/model_optimization/blob/9513796726e72ebdb5b075f5014eb8feae47f3ae
            /model_compression_toolkit/hardware_models/keras_hardware_model/keras_default.py#L39>`_

        Returns:
            A KPI object with total weights parameters sum and max activation tensor.

        Examples:
            Import a Keras model:

            >>> from tensorflow.keras.applications.mobilenet import MobileNet
            >>> model = MobileNet()

            Create a random dataset generator:

            >>> import numpy as np
            >>> def repr_datagen(): return [np.random.random((1,224,224,3))]

            Import mct and call for KPI data calculation:
            >>> import model_compression_toolkit as mct
            >>> kpi_data = keras_kpi_data(model, repr_datagen)

        """

        if not isinstance(quant_config, MixedPrecisionQuantizationConfig):
            Logger.error("KPI data computation can be executed without MixedPrecisionQuantizationConfig object."
                         "Given quant_config is not of type MixedPrecisionQuantizationConfig.")

        fw_impl = KerasImplementation()

        return compute_kpi_data(in_model,
                                representative_data_gen,
                                quant_config,
                                target_platform_capabilities,
                                fw_info,
                                fw_impl)

else:
    # If tensorflow or tensorflow_model_optimization are not installed,
    # we raise an exception when trying to use this function.
    def keras_kpi_data(*args, **kwargs):
        Logger.critical('Installing tensorflow and tensorflow_model_optimization is mandatory '
                        'when using keras_kpi_data. '
                        'Could not find Tensorflow package.')
