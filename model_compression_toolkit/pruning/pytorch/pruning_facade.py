# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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

from typing import Callable, Tuple
from model_compression_toolkit import get_target_platform_capabilities
from model_compression_toolkit.constants import PYTORCH
from model_compression_toolkit.verify_packages import FOUND_TORCH
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import ResourceUtilization
from model_compression_toolkit.core.common.pruning.pruner import Pruner
from model_compression_toolkit.core.common.pruning.pruning_config import PruningConfig
from model_compression_toolkit.core.common.pruning.pruning_info import PruningInfo
from model_compression_toolkit.core.common.quantization.bit_width_config import BitWidthConfig
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import set_quantization_configuration_to_graph
from model_compression_toolkit.core.graph_prep_runner import read_model_to_graph
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework import TargetPlatformCapabilities
from model_compression_toolkit.core.common.quantization.quantization_config import DEFAULTCONFIG
from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL


# Check if PyTorch is available in the environment.
if FOUND_TORCH:
    # Import PyTorch-specific modules from the model compression toolkit.
    from model_compression_toolkit.core.pytorch.back2framework.float_model_builder import FloatPyTorchModelBuilder
    from model_compression_toolkit.core.pytorch.pruning.pruning_pytorch_implementation import \
        PruningPytorchImplementation
    from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
    from torch.nn import Module

    # Set the default Target Platform Capabilities (TPC) for PyTorch.
    DEFAULT_PYOTRCH_TPC = get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL)

    def pytorch_pruning_experimental(model: Module,
                                     target_resource_utilization: ResourceUtilization,
                                     representative_data_gen: Callable,
                                     pruning_config: PruningConfig = PruningConfig(),
                                     target_platform_capabilities: TargetPlatformCapabilities = DEFAULT_PYOTRCH_TPC) -> \
            Tuple[Module, PruningInfo]:
        """
        Perform structured pruning on a Pytorch model to meet a specified target resource utilization.
        This function prunes the provided model according to the target resource utilization by grouping and pruning
        channels based on each layer's SIMD configuration in the Target Platform Capabilities (TPC).
        By default, the importance of each channel group is determined using the Label-Free Hessian
        (LFH) method, assessing each channel's sensitivity to the Hessian of the loss function.
        This pruning strategy considers groups of channels together for a more hardware-friendly
        architecture. The process involves analyzing the model with a representative dataset to
        identify groups of channels that can be removed with minimal impact on performance.

        Notice that the pruned model must be retrained to recover the compressed model's performance.

        Args:
            model (Module): The PyTorch model to be pruned.
            target_resource_utilization (ResourceUtilization): Key Performance Indicators specifying the pruning targets.
            representative_data_gen (Callable): A function to generate representative data for pruning analysis.
            pruning_config (PruningConfig): Configuration settings for the pruning process. Defaults to standard config.
            target_platform_capabilities (TargetPlatformCapabilities): Platform-specific constraints and capabilities.
                Defaults to DEFAULT_PYTORCH_TPC.

        Returns:
            Tuple[Model, PruningInfo]: A tuple containing the pruned Pytorch model and associated pruning information.

        Note:
            The pruned model should be fine-tuned or retrained to recover or improve its performance post-pruning.

        Examples:

            Import MCT:

            >>> import model_compression_toolkit as mct

            Import a Pytorch model:

            >>> from torchvision.models import resnet50, ResNet50_Weights
            >>> model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

            Create a random dataset generator:

            >>> import numpy as np
            >>> def repr_datagen(): yield [np.random.random((1, 3, 224, 224))]

            Define a target resource utilization for pruning.
            Here, we aim to reduce the memory footprint of weights by 50%, assuming the model weights
            are represented in float32 data type (thus, each parameter is represented using 4 bytes):

            >>> dense_nparams = sum(p.numel() for p in model.state_dict().values())
            >>> target_resource_utilization = mct.core.ResourceUtilization(weights_memory=dense_nparams * 4 * 0.5)

            Optionally, define a pruning configuration. num_score_approximations can be passed
            to configure the number of importance scores that will be calculated for each channel.
            A higher value for this parameter yields more precise score approximations but also
            extends the duration of the pruning process:

            >>> pruning_config = mct.pruning.PruningConfig(num_score_approximations=1)

            Perform pruning:

            >>> pruned_model, pruning_info = mct.pruning.pytorch_pruning_experimental(model=model, target_resource_utilization=target_resource_utilization, representative_data_gen=repr_datagen, pruning_config=pruning_config)

        """

        Logger.warning(f"pytorch_pruning_experimental is experimental and is subject to future changes."
                       f"If you encounter an issue, please open an issue in our GitHub "
                       f"project https://github.com/sony/model_optimization")

        # Instantiate the Pytorch framework implementation.
        fw_impl = PruningPytorchImplementation()

        # Convert the original Pytorch model to an internal graph representation.
        float_graph = read_model_to_graph(model,
                                          representative_data_gen,
                                          target_platform_capabilities,
                                          DEFAULT_PYTORCH_INFO,
                                          fw_impl)

        # Apply quantization configuration to the graph. This step is necessary even when not quantizing,
        # as it prepares the graph for the pruning process.
        float_graph_with_compression_config = set_quantization_configuration_to_graph(float_graph,
                                                                                      quant_config=DEFAULTCONFIG,
                                                                                      mixed_precision_enable=False)

        # Create a Pruner object with the graph and configuration.
        pruner = Pruner(float_graph_with_compression_config,
                        DEFAULT_PYTORCH_INFO,
                        fw_impl,
                        target_resource_utilization,
                        representative_data_gen,
                        pruning_config,
                        target_platform_capabilities)

        # Apply the pruning process.
        pruned_graph = pruner.prune_graph()

        # Retrieve pruning information which includes the pruning masks and scores.
        pruning_info = pruner.get_pruning_info()

        # Rebuild the pruned graph back into a trainable Pytorch model.
        pruned_model, _ = FloatPyTorchModelBuilder(graph=pruned_graph).build_model()
        pruned_model.trainable = True

        # Return the pruned model along with its pruning information.
        return pruned_model, pruning_info

else:
    def pytorch_pruning_experimental(*args, **kwargs):
        """
        Raises a critical error if PyTorch is not installed but the pruning function is invoked.

        This function acts as a placeholder to provide a clear error message when PyTorch dependencies are missing,
        indicating that the pruning functionality cannot be used without the PyTorch framework installed.

        Args:
            *args: Variable length argument list, not used.
            **kwargs: Arbitrary keyword arguments, not used.

        Raises:
            CriticalError: Indicates that PyTorch must be installed to use this function.
        """
        Logger.critical("PyTorch must be installed to use 'pytorch_pruning_experimental'. "
                        "The 'torch' package is missing.")  # pragma: no cover
