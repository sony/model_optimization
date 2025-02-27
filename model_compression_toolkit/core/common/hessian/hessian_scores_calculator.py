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

from abc import ABC, abstractmethod
from typing import List, Any

from model_compression_toolkit.constants import HESSIAN_NUM_ITERATIONS
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.hessian import HessianScoresRequest
from model_compression_toolkit.logger import Logger


class HessianScoresCalculator(ABC):
    """
    Abstract base class for computing scores based on the Hessian matrix approximation.

    This class provides a structure for implementing different methods to compute
    scores based on Hessian-approximation according to the provided configuration,
    input images, and other parameters.
    """

    def __init__(self,
                 graph: Graph,
                 input_images: List[Any],
                 fw_impl,
                 hessian_scores_request: HessianScoresRequest,
                 num_iterations_for_approximation: int = HESSIAN_NUM_ITERATIONS):
        """
        Args:
            graph: Computational graph for the float model.
            input_images: List of input images for the computation.
            fw_impl: Framework-specific implementation for Hessian-approximation scores computation.
            hessian_scores_request: Configuration request for which to compute the Hessian-based approximation.
            num_iterations_for_approximation: Number of iterations to use when approximating the Hessian-approximation scores.

        """
        self.graph = graph

        for output_node in graph.get_outputs():
            if not fw_impl.is_output_node_compatible_for_hessian_score_computation(output_node.node):
                Logger.critical(f"All graph outputs must support Hessian score computation. Incompatible node: {output_node.node}, layer type: {output_node.node.type}. Consider disabling Hessian info computation.")

        self.input_images = fw_impl.to_tensor(input_images)
        self.num_iterations_for_approximation = num_iterations_for_approximation

        # Validate representative dataset has same inputs as graph
        if not hessian_scores_request.compute_from_tensors and len(self.input_images) != len(graph.get_inputs()):  # pragma: no cover
            Logger.critical(f"The graph requires {len(graph.get_inputs())} inputs, but the provided representative dataset contains {len(self.input_images)} inputs.")

        self.fw_impl = fw_impl
        self.hessian_request = hessian_scores_request

    @abstractmethod
    def compute(self) -> List[float]:
        """
        Abstract method to compute the scores based on the Hessian-approximation matrix.

        This method should be implemented by subclasses to provide the specific
        computation method for the Hessian-approximation scores.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement compute method.')  # pragma: no cover

    @staticmethod
    def unfold_tensors_list(tensors_to_unfold: Any) -> List[Any]:
        """
        Unfold (flatten) a nested tensors list.
        Given a mixed list of single tensors and nested tensor lists,
        this method returns a flattened list where nested lists are expanded.
        Args:
            tensors_to_unfold: Tensors to unfold.
        Returns:
            A flattened list of tensors.
        """
        unfold_tensors = []
        for tensor in tensors_to_unfold:
            if isinstance(tensor, List):
                unfold_tensors += tensor
            else:
                unfold_tensors.append(tensor)
        return unfold_tensors
