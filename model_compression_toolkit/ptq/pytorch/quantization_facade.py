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
import copy

from typing import Callable

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.visualization.tensorboard_writer import init_tensorboard_writer
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.constants import PYTORCH, FOUND_TORCH
from model_compression_toolkit.target_platform_capabilities.target_platform import TargetPlatformCapabilities
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import ResourceUtilization
from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.runner import core_runner
from model_compression_toolkit.ptq.runner import ptq_runner
from model_compression_toolkit.core.analyzer import analyzer_model_quantization
from model_compression_toolkit.core.common.quantization.quantize_graph_weights import quantize_graph_weights
from model_compression_toolkit.metadata import get_versions_dict


if FOUND_TORCH:
    from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
    from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
    from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL
    from torch.nn import Module
    from model_compression_toolkit.exporter.model_wrapper.pytorch.builder.fully_quantized_model_builder import get_exportable_pytorch_model
    from model_compression_toolkit import get_target_platform_capabilities
    from mct_quantizers.pytorch.metadata import add_metadata

    DEFAULT_PYTORCH_TPC = get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL)

    def pytorch_post_training_quantization(in_module: Module,
                                           representative_data_gen: Callable,
                                           target_resource_utilization: ResourceUtilization = None,
                                           core_config: CoreConfig = CoreConfig(),
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
            target_resource_utilization (ResourceUtilization): ResourceUtilization object to limit the search of the mixed-precision configuration as desired.
            core_config (CoreConfig): Configuration object containing parameters of how the model should be quantized, including mixed precision parameters.
            target_platform_capabilities (TargetPlatformCapabilities): TargetPlatformCapabilities to optimize the PyTorch model according to.

        Returns:
            A quantized module and information the user may need to handle the quantized module.

        Examples:

            Import a Pytorch module:

            >>> from torchvision import models
            >>> module = models.mobilenet_v2()

            Create a random dataset generator, for required number of calibration iterations (num_calibration_batches):
            In this example a random dataset of 10 batches each containing 4 images is used.

            >>> import numpy as np
            >>> num_calibration_batches = 10
            >>> def repr_datagen():
            >>>     for _ in range(num_calibration_batches):
            >>>         yield [np.random.random((4, 3, 224, 224))]

            Import MCT and pass the module with the representative dataset generator to get a quantized module
            Set number of clibration iterations to 1:

            >>> import model_compression_toolkit as mct
            >>> quantized_module, quantization_info = mct.ptq.pytorch_post_training_quantization(module, repr_datagen)

        """

        fw_info = DEFAULT_PYTORCH_INFO

        if core_config.mixed_precision_enable:
            if not isinstance(core_config.mixed_precision_config, MixedPrecisionQuantizationConfig):
                Logger.critical("Given quantization config to mixed-precision facade is not of type "
                                "MixedPrecisionQuantizationConfig. Please use "
                                "pytorch_post_training_quantization API, or pass a valid mixed precision "
                                "configuration.")  # pragma: no cover

        tb_w = init_tensorboard_writer(fw_info)

        fw_impl = PytorchImplementation()

        # Ignore hessian info service as it is not used here yet.
        tg, bit_widths_config, _ = core_runner(in_model=in_module,
                                               representative_data_gen=representative_data_gen,
                                               core_config=core_config,
                                               fw_info=fw_info,
                                               fw_impl=fw_impl,
                                               tpc=target_platform_capabilities,
                                               target_resource_utilization=target_resource_utilization,
                                               tb_w=tb_w)

        # At this point, tg is a graph that went through substitutions (such as BN folding) and is
        # ready for quantization (namely, it holds quantization params, etc.) but the weights are
        # not quantized yet. For this reason, we use it to create a graph that acts as a "float" graph
        # for things like similarity analyzer (because the quantized and float graph should have the same
        # architecture to find the appropriate compare points for similarity computation).
        similarity_baseline_graph = copy.deepcopy(tg)

        graph_with_stats_correction = ptq_runner(tg,
                                                 representative_data_gen,
                                                 core_config,
                                                 fw_info,
                                                 fw_impl,
                                                 tb_w)

        if core_config.debug_config.analyze_similarity:
            quantized_graph = quantize_graph_weights(graph_with_stats_correction)
            analyzer_model_quantization(representative_data_gen,
                                        tb_w,
                                        similarity_baseline_graph,
                                        quantized_graph,
                                        fw_impl,
                                        fw_info)

        exportable_model, user_info = get_exportable_pytorch_model(graph_with_stats_correction)
        if target_platform_capabilities.tp_model.add_metadata:
            exportable_model = add_metadata(exportable_model, get_versions_dict(target_platform_capabilities))
        return exportable_model, user_info


else:
    # If torch is not installed,
    # we raise an exception when trying to use these functions.
    def pytorch_post_training_quantization(*args, **kwargs):
        Logger.critical("PyTorch must be installed to use 'pytorch_post_training_quantization_experimental'. "
                        "The 'torch' package is missing.")  # pragma: no cover
