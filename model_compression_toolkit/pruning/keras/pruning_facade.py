# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from model_compression_toolkit.constants import TENSORFLOW, FOUND_TF
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.pruning.pruner import Pruner
from model_compression_toolkit.core.common.pruning.pruning_config import PruningConfig
from model_compression_toolkit.core.common.pruning.pruning_info import PruningInfo
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.core.graph_prep_runner import read_model_to_graph
from model_compression_toolkit.core.keras.back2framework.float_model_builder import FloatKerasModelBuilder
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework import \
    TargetPlatformCapabilities

if FOUND_TF:
    from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
    from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
    from tensorflow.keras.models import Model
    from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL

    from model_compression_toolkit.core.common.quantization.quantization_config import DEFAULTCONFIG

    DEFAULT_KERAS_TPC = get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)


    def keras_pruning_experimental(model: Model,
                                   target_kpi: KPI,
                                   representative_data_gen: Callable,
                                   pruning_config: PruningConfig = PruningConfig(),
                                   target_platform_capabilities: TargetPlatformCapabilities = DEFAULT_KERAS_TPC) -> \
            Tuple[Model, PruningInfo]:
        """
        Perform experimental pruning on a Keras model to meet a specified target KPI.

        Args:
            model (Model): The original Keras model to be pruned.
            target_kpi (KPI): The target Key Performance Indicators to be achieved through pruning.
            representative_data_gen (Callable): A function to generate representative data for pruning analysis.
            pruning_config (PruningConfig): Configuration settings for the pruning process. Defaults to standard config.
            target_platform_capabilities (TargetPlatformCapabilities): Platform-specific constraints and capabilities.
                Defaults to DEFAULT_KERAS_TPC.

        Returns:
            Tuple[Model, PruningInfo]: A tuple containing the pruned Keras model and associated pruning information.
        """

        # Instantiate the Keras framework implementation.
        fw_impl = KerasImplementation()

        # Convert the original Keras model to an internal graph representation.
        float_graph = read_model_to_graph(model,
                                          representative_data_gen,
                                          target_platform_capabilities,
                                          DEFAULT_KERAS_INFO,
                                          fw_impl)

        # Apply quantization configuration to the graph. This step is necessary even when not quantizing,
        # as it prepares the graph for compression.
        float_graph_with_compression_config = set_quantization_configuration_to_graph(float_graph,
                                                                                      quant_config=DEFAULTCONFIG,
                                                                                      mixed_precision_enable=False)

        # Create a Pruner object with the graph and configuration.
        pruner = Pruner(float_graph_with_compression_config,
                        DEFAULT_KERAS_INFO,
                        fw_impl,
                        target_kpi,
                        representative_data_gen,
                        pruning_config,
                        target_platform_capabilities)

        # Apply the pruning process.
        pruned_graph = pruner.get_pruned_graph() #TODO:rename

        # Retrieve pruning information which includes the pruning masks and scores.
        pruning_info = pruner.get_pruning_info()

        # Rebuild the pruned graph back into a trainable Keras model.
        pruned_model, _ = FloatKerasModelBuilder(graph=pruned_graph).build_model()

        # Return the pruned model along with its pruning information.
        return pruned_model, pruning_info



else:
    # If tensorflow is not installed,
    # we raise an exception when trying to use these functions.
    def keras_pruning_experimental(*args, **kwargs):
        Logger.critical('Installing tensorflow is mandatory '
                        'when using keras_pruning_experimental. '
                        'Could not find Tensorflow package.')  # pragma: no cover
