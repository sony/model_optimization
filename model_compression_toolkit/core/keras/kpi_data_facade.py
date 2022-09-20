# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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

from model_compression_toolkit import MixedPrecisionQuantizationConfig, CoreConfig, MixedPrecisionQuantizationConfigV2
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.constants import TENSORFLOW
from model_compression_toolkit.core.common.target_platform import TargetPlatformCapabilities
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi_data import compute_kpi_data
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    DEFAULT_MIXEDPRECISION_CONFIG

import importlib


if importlib.util.find_spec("tensorflow") is not None\
        and importlib.util.find_spec("tensorflow_model_optimization") is not None:
    from model_compression_toolkit.core.keras.constants import DEFAULT_TP_MODEL
    from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
    from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
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
        Builds the computation graph from the given model and target platform modeling, and uses it to compute the KPI data.

        Args:
            in_model (Model): Keras model to quantize.
            representative_data_gen (Callable): Dataset used for calibration.
            quant_config (MixedPrecisionQuantizationConfig): MixedPrecisionQuantizationConfig containing parameters of how the model should be quantized.
            fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.). `Default Keras info <https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/keras/default_framework_info.py>`_
            target_platform_capabilities (TargetPlatformCapabilities): TargetPlatformCapabilities to optimize the Keras model according to. `Default Keras TPC <https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/tpc_models/keras_tp_models/keras_default.py>`_

        Returns:
            A KPI object with total weights parameters sum, max activation tensor and total kpi.

        Examples:

            Import a Keras model:

            >>> from tensorflow.keras.applications.mobilenet import MobileNet
            >>> model = MobileNet()

            Create a random dataset generator:

            >>> import numpy as np
            >>> def repr_datagen(): return [np.random.random((1,224,224,3))]

            Import MCT and call for KPI data calculation:

            >>> import model_compression_toolkit as mct
            >>> kpi_data = mct.keras_kpi_data(model, repr_datagen)


        """

        if not isinstance(quant_config, MixedPrecisionQuantizationConfig):
            Logger.error("KPI data computation can't be executed without MixedPrecisionQuantizationConfig object."
                         "Given quant_config is not of type MixedPrecisionQuantizationConfig.")

        fw_impl = KerasImplementation()

        quantization_config, mp_config = quant_config.separate_configs()
        core_config = CoreConfig(quantization_config=quantization_config,
                                 mixed_precision_config=mp_config)

        return compute_kpi_data(in_model,
                                representative_data_gen,
                                core_config,
                                target_platform_capabilities,
                                fw_info,
                                fw_impl)


    def keras_kpi_data_experimental(in_model: Model,
                                    representative_data_gen: Callable,
                                    core_config: CoreConfig,
                                    fw_info: FrameworkInfo = DEFAULT_KERAS_INFO,
                                    target_platform_capabilities: TargetPlatformCapabilities = KERAS_DEFAULT_TPC) -> KPI:
        """
        Computes KPI data that can be used to calculate the desired target KPI for mixed-precision quantization.
        Builds the computation graph from the given model and hw modeling, and uses it to compute the KPI data.

        Args:
            in_model (Model): Keras model to quantize.
            representative_data_gen (Callable): Dataset used for calibration.
            core_config (CoreConfig): CoreConfig containing parameters for quantization and mixed precision of how the model should be quantized.
            fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.). `Default Keras info <https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/keras/default_framework_info.py>`_
            target_platform_capabilities (TargetPlatformCapabilities): TargetPlatformCapabilities to optimize the Keras model according to. `Default Keras TPC <https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/tpc_models/keras_tp_models/keras_default.py>`_

        Returns:

            A KPI object with total weights parameters sum and max activation tensor.

        Examples:

            Import a Keras model:

            >>> from tensorflow.keras.applications.mobilenet import MobileNet
            >>> model = MobileNet()

            Create a random dataset generator:

            >>> import numpy as np
            >>> def repr_datagen(): return [np.random.random((1,224,224,3))]

            Import MCT and call for KPI data calculation:

            >>> import model_compression_toolkit as mct
            >>> kpi_data = mct.keras_kpi_data(model, repr_datagen)

        """

        if not isinstance(core_config.mixed_precision_config, MixedPrecisionQuantizationConfigV2):
            Logger.error("KPI data computation can't be executed without MixedPrecisionQuantizationConfigV2 object."
                         "Given quant_config is not of type MixedPrecisionQuantizationConfigV2.")

        fw_impl = KerasImplementation()

        return compute_kpi_data(in_model,
                                representative_data_gen,
                                core_config,
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


    def keras_kpi_data_experimental(*args, **kwargs):
        Logger.critical('Installing tensorflow and tensorflow_model_optimization is mandatory '
                        'when using keras_kpi_data. '
                        'Could not find Tensorflow package.')
