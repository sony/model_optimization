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

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.constants import PYTORCH, FOUND_TORCH
from model_compression_toolkit.core.common.target_platform import TargetPlatformCapabilities
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit import CoreConfig
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfigV2
from model_compression_toolkit.core.runner import core_runner, _init_tensorboard_writer
from model_compression_toolkit.ptq.runner import ptq_runner
from model_compression_toolkit.core.exporter import export_model
from model_compression_toolkit.core.analyzer import analyzer_model_quantization


if FOUND_TORCH:
    from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
    from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
    from model_compression_toolkit.core.pytorch.constants import DEFAULT_TP_MODEL
    from torch.nn import Module

    from model_compression_toolkit import get_target_platform_capabilities
    DEFAULT_PYTORCH_TPC = get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL)


    def pytorch_post_training_quantization_experimental(in_module: Module,
                                                        representative_data_gen: Callable,
                                                        target_kpi: KPI = None,
                                                        core_config: CoreConfig = CoreConfig(),
                                                        fw_info: FrameworkInfo = DEFAULT_PYTORCH_INFO,
                                                        target_platform_capabilities: TargetPlatformCapabilities = DEFAULT_PYTORCH_TPC):
        """
        Quantize a trained Pytorch module using post-training quantization.
        By default, the module is quantized using a symmetric constraint quantization thresholds
        (power of two) as defined in the default TargetPlatformCapabilities.
        The module is first optimized using several transformations (e.g. BatchNormalization folding to
        preceding layers). Then, using a given dataset, statistics (e.g. min/max, histogram, etc.) are
        being collected for each layer's output (and input, depends on the quantization configuration).
        Thresholds are then being calculated using the collected statistics and the module is quantized
        (both coefficients and activations by default).
        If gptq_config is passed, the quantized weights are optimized using gradient based post
        training quantization by comparing points between the float and quantized modules, and minimizing the
        observed loss.

        Args:
            in_module (Module): Pytorch module to quantize.
            representative_data_gen (Callable): Dataset used for calibration.
            target_kpi (KPI): KPI object to limit the search of the mixed-precision configuration as desired.
            core_config (CoreConfig): Configuration object containing parameters of how the model should be quantized, including mixed precision parameters.
            fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.). `Default PyTorch info <https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/pytorch/default_framework_info.py>`_
            target_platform_capabilities (TargetPlatformCapabilities): TargetPlatformCapabilities to optimize the PyTorch model according to. `Default PyTorch TPC <https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/tpc_models/pytorch_tp_models/pytorch_default.py>`_


        Returns:
            A quantized module and information the user may need to handle the quantized module.

        Examples:

            Import a Pytorch module:

            >>> import torchvision.models.mobilenet_v2 as models
            >>> module = models.mobilenet_v2()

            Create a random dataset generator:

            >>> import numpy as np
            >>> def repr_datagen(): return [np.random.random((1,224,224,3))]

            Import mct and pass the module with the representative dataset generator to get a quantized module:

            >>> import model_compression_toolkit as mct
            >>> quantized_module, quantization_info = mct.pytorch_post_training_quantization(module, repr_datagen)

        """

        if core_config.mixed_precision_enable:
            if not isinstance(core_config.mixed_precision_config, MixedPrecisionQuantizationConfigV2):
                common.Logger.error("Given quantization config to mixed-precision facade is not of type "
                                    "MixedPrecisionQuantizationConfigV2. Please use pytorch_post_training_quantization API,"
                                    "or pass a valid mixed precision configuration.")

            common.Logger.info("Using experimental mixed-precision quantization. "
                               "If you encounter an issue please file a bug.")

        tb_w = _init_tensorboard_writer(fw_info)

        fw_impl = PytorchImplementation()

        tg, bit_widths_config = core_runner(in_model=in_module,
                                            representative_data_gen=representative_data_gen,
                                            core_config=core_config,
                                            fw_info=fw_info,
                                            fw_impl=fw_impl,
                                            tpc=target_platform_capabilities,
                                            target_kpi=target_kpi,
                                            tb_w=tb_w)

        tg = ptq_runner(tg, fw_info, fw_impl, tb_w)

        if core_config.debug_config.analyze_similarity:
            analyzer_model_quantization(representative_data_gen, tb_w, tg, fw_impl, fw_info)

        quantized_model, user_info = export_model(tg, fw_info, fw_impl, tb_w, bit_widths_config)

        return quantized_model, user_info

else:
    # If torch is not installed,
    # we raise an exception when trying to use these functions.
    def pytorch_post_training_quantization_experimental(*args, **kwargs):
        Logger.critical('Installing Pytorch is mandatory '
                        'when using pytorch_post_training_quantization_experimental. '
                        'Could not find the torch package.')
