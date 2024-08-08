#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo

from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.common.quantization.bit_width_config import BitWidthConfig
from model_compression_toolkit.core.common.quantization.quantization_config import DEFAULTCONFIG

from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner
from typing import Any, Callable

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.target_platform_capabilities.target_platform import TargetPlatformCapabilities


class ModelFoldingUtils:
    """
    Utility class for handling model folding operations such as batch normalization (BN) folding,
    residual collapsing, and other graph optimizations.
    """

    def __init__(self,
                 fw_info: FrameworkInfo,
                 fw_impl: FrameworkImplementation,
                 fw_default_tpc: TargetPlatformCapabilities):
        """
        Initialize the ModelFoldingUtils class with framework-specific information, implementation details,
        and default TPC.

        Args:
            fw_info: Framework-specific information.
            fw_impl: Implementation functions for the framework.
            fw_default_tpc: Default target platform capabilities for the handled framework.
        """
        self.fw_info = fw_info
        self.fw_impl = fw_impl
        self.fw_default_tpc = fw_default_tpc

    def create_float_folded_model(self, float_model: Any, representative_dataset: Any = None) -> Any:
        """
        Create folded version of the model like MCT does (bn folding, residual collapsing, etc.).
        This is needed since we need the models we compare to have the same architecture for
        comparing tensors in different points of the models.

        Args:
            float_model: The floating-point model to be folded.
            representative_dataset: A callable for generating representative data.

        Returns:
            The folded floating-point model.

        """
        float_graph = self.create_float_folded_graph(model=float_model,
                                                     repr_dataset=representative_dataset)
        float_folded_model, _ = self.fw_impl.model_builder(
            float_graph,
            mode=ModelBuilderMode.FLOAT,
            append2output=None,
            fw_info=self.fw_info
        )
        return float_folded_model

    def create_float_folded_graph(self, model: Any, repr_dataset: Callable) -> Graph:
        """
        Create a folded graph for the float model. This process involves
        graph optimizations similar to those applied during quantization (e.g., batch normalization folding,
        residual collapsing).

        Args:
            model: The floating-point model to be folded into a graph.
            repr_dataset: A callable that generates representative data.

        Returns:
            The folded graph.
        """
        # TODO:
        # Consider simplifying graph_preparation_runner by extracting relevant parts to a separate method in MCT.
        #
        # Issues:
        # 1. The quantization config affects how the optimized graph looks (e.g., collapsing).
        # 2. The back2fw function requires quantization info even for float models.
        #
        # Future Considerations:
        # - Remove quantization config parts related to graph optimizations.
        # - Update back2fw to handle float models without needing quantization info.
        graph = graph_preparation_runner(in_model=model,
                                         representative_data_gen=repr_dataset,
                                         fw_impl=self.fw_impl,
                                         fw_info=self.fw_info,
                                         quantization_config=DEFAULTCONFIG,
                                         tpc=self.fw_default_tpc)
        return graph
