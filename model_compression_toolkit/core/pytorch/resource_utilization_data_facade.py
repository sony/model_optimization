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

from typing import Callable, Union

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.constants import PYTORCH
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformCapabilities
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import ResourceUtilization
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization_data import compute_resource_utilization_data
from model_compression_toolkit.core.common.quantization.core_config import CoreConfig
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import MixedPrecisionQuantizationConfig
from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL
from model_compression_toolkit.target_platform_capabilities.tpc_io_handler import load_target_platform_capabilities
from model_compression_toolkit.verify_packages import FOUND_TORCH

if FOUND_TORCH:
    from model_compression_toolkit.core.pytorch.default_framework_info import set_pytorch_info
    from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
    from torch.nn import Module
    from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
        AttachTpcToPytorch

    from model_compression_toolkit import get_target_platform_capabilities

    PYTORCH_DEFAULT_TPC = get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL)


    @set_pytorch_info
    def pytorch_resource_utilization_data(in_model: Module,
                                          representative_data_gen: Callable,
                                          core_config: CoreConfig = CoreConfig(),
                                          target_platform_capabilities: Union[TargetPlatformCapabilities, str] = PYTORCH_DEFAULT_TPC
                                          ) -> ResourceUtilization:
        """
        Computes resource utilization data that can be used to calculate the desired target resource utilization for mixed-precision quantization.
        Builds the computation graph from the given model and target platform capabilities, and uses it to compute the resource utilization data.

        Args:
            in_model (Model): PyTorch model to quantize.
            representative_data_gen (Callable): Dataset used for calibration.
            core_config (CoreConfig): CoreConfig containing parameters for quantization and mixed precision
            target_platform_capabilities (Union[TargetPlatformCapabilities, str]): FrameworkQuantizationCapabilities to optimize the PyTorch model according to.

        Returns:

            A ResourceUtilization object with total weights parameters sum and max activation tensor.

        Examples:

            Import a Pytorch model:

            >>> from torchvision import models
            >>> module = models.mobilenet_v2()

            Create a random dataset generator:

            >>> import numpy as np
            >>> def repr_datagen(): yield [np.random.random((1, 3, 224, 224))]

            Import mct and call for resource utilization data calculation:

            >>> import model_compression_toolkit as mct
            >>> ru_data = mct.core.pytorch_resource_utilization_data(module, repr_datagen)

        """

        if not isinstance(core_config.mixed_precision_config, MixedPrecisionQuantizationConfig):
            Logger.critical("Resource utilization data computation requires a MixedPrecisionQuantizationConfig object. "
                            "The provided 'mixed_precision_config' is not of this type.")

        fw_impl = PytorchImplementation()

        target_platform_capabilities = load_target_platform_capabilities(target_platform_capabilities)
        # Attach tpc model to framework
        attach2pytorch = AttachTpcToPytorch()
        target_platform_capabilities = (
            attach2pytorch.attach(target_platform_capabilities,
                                  custom_opset2layer=core_config.quantization_config.custom_tpc_opset_to_layer))

        return compute_resource_utilization_data(in_model,
                                                 representative_data_gen,
                                                 core_config,
                                                 target_platform_capabilities,
                                                 fw_impl)

else:
    # If torch is not installed,
    # we raise an exception when trying to use this function.
    def pytorch_resource_utilization_data(*args, **kwargs):
        Logger.critical("PyTorch must be installed to use 'pytorch_resource_utilization_data'. "
                        "The 'torch' package is missing.")  # pragma: no cover

