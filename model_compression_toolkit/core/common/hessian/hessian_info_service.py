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

from functools import partial
from typing import Callable, List

from model_compression_toolkit.constants import HESSIAN_NUM_ITERATIONS
from model_compression_toolkit.core.common.hessian.trace_hessian_request import TraceHessianRequest
from model_compression_toolkit.logger import Logger


class HessianInfoService:
    """
    A service to manage, store, and compute approximation of the Hessian-based information.

    This class provides functionalities to compute approximation based on the Hessian matrix based
    on the different parameters (such as number of iterations for approximating the info)
    and input images (from representative_dataset).
    It also offers cache management capabilities for efficient computation and retrieval.

    Note:
    - The Hessian provides valuable information about the curvature of the loss function.
    - Computation can be computationally heavy and time-consuming.
    - The computed trace is an approximation.
    """

    def __init__(self,
                 graph,
                 representative_dataset: Callable,
                 fw_impl,
                 num_iterations_for_approximation: int = HESSIAN_NUM_ITERATIONS
                 ):
        """

        Args:
            graph: Float graph.
            representative_dataset: A callable that provides a dataset for sampling.
            fw_impl: Framework-specific implementation for trace Hessian approximation computation.
        """
        self.graph = graph

        # Create a representative_data_gen with batch size of 1
        self.representative_dataset = partial(self._sample_single_representative_dataset,
                                              representative_dataset=representative_dataset)

        self.fw_impl = fw_impl
        self.num_iterations_for_approximation = num_iterations_for_approximation

        self.trace_hessian_request_to_score_list = {}

    def _sample_single_representative_dataset(self, representative_dataset: Callable):
        """
        Get a single sample (namely, batch size of 1) from a representative dataset.

        Args:
            representative_dataset: Callable which returns the representative dataset at any batch size.

        Returns: List of inputs from representative_dataset where each sample has a batch size of 1.
        """
        images = next(representative_dataset())
        if not isinstance(images, list):
            Logger.critical(f'Expected images to be a list; found type: {type(images)}.')

        # Ensure each image is a single sample, if not, take the first sample
        return [image[0:1, ...] if image.shape[0] != 1 else image for image in images]

    def _clear_saved_hessian_info(self):
        """Clears the saved info approximations."""
        self.trace_hessian_request_to_score_list={}

    def count_saved_info_of_request(self, hessian_request:TraceHessianRequest) -> int:
        """
        Counts the saved approximations of Hessian info (traces, for now) for a specific request.
        If some approximations were computed for this request before, the amount of approximations (per image)
        will be returned. If not, zero is returned.

        Args:
            hessian_request: The request configuration for which to count the saved data.

        Returns:
            Number of saved approximations for the given request.
        """
        # Replace request of a reused target node with a request of the 'reuse group'.
        if hessian_request.target_node.reuse_group:
            hessian_request = self._get_request_of_reuse_group(hessian_request)

        # Check if the request is in the saved info and return its count, otherwise return 0
        return len(self.trace_hessian_request_to_score_list.get(hessian_request, []))


    def compute(self, trace_hessian_request:TraceHessianRequest):
        """
        Computes an approximation of the trace of the Hessian based on the
        provided request configuration and stores it in the cache.

        Args:
            trace_hessian_request: Configuration for which to compute the approximation.
        """
        Logger.debug(f"Computing Hessian-trace approximation for a node {trace_hessian_request.target_node}.")

        # Sample images for the computation
        images = self.representative_dataset()

        # Get the framework-specific calculator for trace Hessian approximation
        fw_hessian_calculator = self.fw_impl.get_trace_hessian_calculator(graph=self.graph,
                                                                          input_images=images,
                                                                          trace_hessian_request=trace_hessian_request,
                                                                          num_iterations_for_approximation=self.num_iterations_for_approximation)

        # Compute the approximation
        trace_hessian = fw_hessian_calculator.compute()

        # Store the computed approximation in the saved info
        if trace_hessian_request in self.trace_hessian_request_to_score_list:
            self.trace_hessian_request_to_score_list[trace_hessian_request].append(trace_hessian)
        else:
            self.trace_hessian_request_to_score_list[trace_hessian_request] = [trace_hessian]



    def fetch_hessian(self,
                      trace_hessian_request:
                      TraceHessianRequest, required_size: int) -> List[List[float]]:
        """
        Fetches the computed approximations of the trace of the Hessian for the given 
        request and required size.

        Args:
            trace_hessian_request: Configuration for which to fetch the approximation.
            required_size: Number of approximations required.

        Returns:
            List[List[float]]: List of computed approximations.
            The outer list is per image (thus, has the length as required_size).
            The inner list length dependent on the granularity (1 for per-tensor, 
            OC for per-output-channel when the requested node has OC output-channels, etc.)
        """
        if required_size==0:
            return []

        Logger.info(f"\nEnsuring {required_size} Hessian-trace approximation for node {trace_hessian_request.target_node}.")

        # Replace request of a reused target node with a request of the 'reuse group'.
        if trace_hessian_request.target_node.reuse_group:
            trace_hessian_request = self._get_request_of_reuse_group(trace_hessian_request)

        # Ensure the saved info has the required number of approximations
        self._populate_saved_info_to_size(trace_hessian_request, required_size)

        # Return the saved approximations for the given request
        return self.trace_hessian_request_to_score_list[trace_hessian_request]

    def _get_request_of_reuse_group(self, trace_hessian_request: TraceHessianRequest):
        """
        For each reused group we compute and fetch its members using a single request.
        This method creates and returns a request for the reused group the node is in.

        Args:
            trace_hessian_request: Request to fetch its node's reused group request.

        Returns:
            TraceHessianRequest for all nodes in the reused group.
        """
        father_nodes = [n for n in self.graph.nodes if not n.reuse and n.reuse_group==trace_hessian_request.target_node.reuse_group]
        if len(father_nodes)!=1:
            Logger.critical(f"Expected a single non-reused node in the reused group, but found {len(father_nodes)}.")
        reused_group_request = TraceHessianRequest(target_node=father_nodes[0],
                                                   granularity=trace_hessian_request.granularity,
                                                   mode=trace_hessian_request.mode)
        return reused_group_request


    def _populate_saved_info_to_size(self,
                                     trace_hessian_request: TraceHessianRequest,
                                     required_size: int):
        """
        Ensures that the saved info has the required size of trace Hessian approximations for the given request.

        Args:
            trace_hessian_request: Configuration for which to ensure the saved info size.
            required_size: Required number of trace Hessian approximations.
        """
        # Get the current number of saved approximations for the request
        current_existing_hessians = self.count_saved_info_of_request(trace_hessian_request)

        Logger.info(
            f"Found {current_existing_hessians} Hessian-trace approximations for node {trace_hessian_request.target_node}."
            f" {required_size - current_existing_hessians} approximations left to compute...")

        # Compute the required number of approximations to meet the required size
        for _ in range(required_size - current_existing_hessians):
            self.compute(trace_hessian_request)


