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
from collections.abc import Iterable

import numpy as np
from functools import partial
from typing import Callable, List, Dict, Any

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

        self.representative_dataset_gen = representative_dataset
        # self.representative_dataset = partial(self._sample_batch_representative_dataset,
        #                                       num_inputs=len(self.graph.input_nodes),
        #                                       representative_dataset=representative_dataset)

        # self.representative_dataset = partial(self._sample_batch_representative_dataset,
        #                                       num_inputs=len(self.graph.input_nodes))

        self.fw_impl = fw_impl
        self.num_iterations_for_approximation = num_iterations_for_approximation

        self.trace_hessian_request_to_score_list = {}

    def _sample_batch_representative_dataset(self,
                                             representative_dataset: Iterable,
                                             num_hessian_samples: int,
                                             num_inputs: int,
                                             batch_size: int = 1,
                                             last_iter_remain_samples: List[List[np.ndarray]] = None):
        """
        Get a single sample (namely, batch size of 1) from a representative dataset.

        Args:
            representative_dataset: Callable which returns the representative dataset at any batch size.

        Returns: List of inputs from representative_dataset where each sample has a batch size of 1.
        """

        # hessian_samples = []
        # # Collect the requested number of samples from the representative dataset
        # # for inp_idx in range(num_inputs):
        # for batch in representative_dataset:
        #
        #     if not isinstance(batch, list):
        #         Logger.critical(f'Expected images to be a list; found type: {type(batch)}.')
        #
        #     # TODO: handle model with multiple inputs. Also, if returning more than one this is because we need more
        #     #  samples than batch size not for multiple inputs, so need to fix that (ActivationHessianTraceAdvanceModelTest) failing on that
        #
        #     batch = batch[0]
        #
        #     # Compute number of missing samples to get to the requested amount from the current batch
        #     num_missing = min(num_hessian_samples - len(hessian_samples), batch.shape[0])
        #     # Append each sample separately
        #     samples = [s for s in batch[0:num_missing, ...]]
        #     hessian_samples += [sample.reshape(1, *sample.shape) for sample in samples]
        #
        #     if len(hessian_samples) > num_hessian_samples:
        #         Logger.critical(f"Requested {num_hessian_samples} samples for computing Hessian approximation but "
        #                         f"{len(hessian_samples)} were collected.")  # pragma: no cover
        #     elif len(hessian_samples) == num_hessian_samples:
        #         # Collected enough samples, constructing a dataset with the requested batch size
        #         hessian_samples = np.concatenate(hessian_samples, axis=0)
        #         num_collected_samples = hessian_samples.shape[0]
        #         hessian_samples = np.split(hessian_samples,
        #                                    num_collected_samples // min(num_collected_samples, batch_size))
        #         return hessian_samples
        #
        # Logger.critical(
        #     f"Not enough samples in the provided representative dataset to compute Hessian approximation on "
        #     f"{num_hessian_samples} samples.")


# --------------------------------------------
        # for batch in representative_dataset:
        all_inp_hessian_samples = [[] for _ in range(num_inputs)]
        # Collect the requested number of samples from the representative dataset
        # for batch in representative_dataset():
        for batch in representative_dataset:
            if not isinstance(batch, list):
                Logger.critical(f'Expected images to be a list; found type: {type(batch)}.')
            all_inp_remaining_samples = [[] for _ in range(num_inputs)]
            for inp_idx in range(len(batch)):
                inp_batch = batch[inp_idx]

                if last_iter_remain_samples is not None and len(last_iter_remain_samples[inp_idx]):
                    # some samples remained from last batch of last computation iteration -
                    # include them in the current batch
                    inp_batch = np.concatenate((inp_batch, last_iter_remain_samples[inp_idx]))

                # Compute number of missing samples to get to the requested amount from the current batch
                num_missing = min(num_hessian_samples - len(all_inp_hessian_samples[inp_idx]), inp_batch.shape[0])
                # Append each sample separately
                samples = [s for s in inp_batch[0:num_missing, ...]]
                # hessian_samples += [sample.reshape(1, *sample.shape) for sample in samples]
                remaining_samples = [s for s in inp_batch[num_missing:, ...]]

                all_inp_hessian_samples[inp_idx] += [sample.reshape(1, *sample.shape) for sample in samples]
                all_inp_remaining_samples[inp_idx] += (remaining_samples)

            if len(all_inp_hessian_samples[0]) > num_hessian_samples:
                Logger.critical(f"Requested {num_hessian_samples} samples for computing Hessian approximation but "
                                f"{len(all_inp_hessian_samples[0])} were collected.")  # pragma: no cover
            elif len(all_inp_hessian_samples[0]) == num_hessian_samples:
                # Collected enough samples, constructing a dataset with the requested batch size
                hessian_samples_for_input = []
                for inp_samples in all_inp_hessian_samples:
                    inp_samples = np.concatenate(inp_samples, axis=0)
                    num_collected_samples = inp_samples.shape[0]
                    inp_samples = np.split(inp_samples,
                                           num_collected_samples // min(num_collected_samples, num_hessian_samples))
                    hessian_samples_for_input.append(inp_samples[0])

                return hessian_samples_for_input, all_inp_remaining_samples
        Logger.critical(
            f"Not enough samples in the provided representative dataset to compute Hessian approximation on "
            f"{num_hessian_samples} samples.")

    def _clear_saved_hessian_info(self):
        """Clears the saved info approximations."""
        self.trace_hessian_request_to_score_list={}

    def count_saved_info_of_request(self, hessian_request: TraceHessianRequest) -> Dict:
        """
        Counts the saved approximations of Hessian info (traces, for now) for a specific request.
        If some approximations were computed for this request before, the amount of approximations (per image)
        will be returned. If not, zero is returned.

        Args:
            hessian_request: The request configuration for which to count the saved data.

        Returns:
            Number of saved approximations for the given request.
        """

        per_node_counter = {}

        for n in hessian_request.target_nodes:
            if n.reuse:
                # Reused nodes supposed to have been replaced with a reuse_group
                # representing node before calling this method.
                Logger.critical(f"Expecting the Hessian request to include only non-reused nodes at this point, "
                                f"but found node {n.name} with 'reuse' status.")
            # Check if the request for this node is in the saved info and store its count, otherwise store 0
            per_node_counter[n] = len(self.trace_hessian_request_to_score_list.get(hessian_request, []))

        return per_node_counter

    def compute(self, trace_hessian_request: TraceHessianRequest, representative_dataset, size_to_compute: int,
                batch_size = 1, last_iter_remain_samples=None):
        """
        Computes an approximation of the trace of the Hessian based on the
        provided request configuration and stores it in the cache.

        Args:
            trace_hessian_request: Configuration for which to compute the approximation.
        """
        Logger.debug(f"Computing Hessian-trace approximation for nodes {trace_hessian_request.target_nodes}.")

        # images = self.representative_dataset(num_hessian_samples=size_to_compute, batch_size=batch_size)
        images, next_iter_remain_samples = representative_dataset(num_hessian_samples=size_to_compute,
                                                                  batch_size=batch_size,
                                                                  last_iter_remain_samples=last_iter_remain_samples)

        # Get the framework-specific calculator for trace Hessian approximation
        fw_hessian_calculator = self.fw_impl.get_trace_hessian_calculator(graph=self.graph,
                                                                          input_images=images,
                                                                          trace_hessian_request=trace_hessian_request,
                                                                          num_iterations_for_approximation=self.num_iterations_for_approximation)

        trace_hessian = fw_hessian_calculator.compute()

        # Store the computed approximation in the saved info
        topo_sorted_nodes_names = [x.name for x in self.graph.get_topo_sorted_nodes()]
        sorted_target_nodes = sorted(trace_hessian_request.target_nodes,
                                     key=lambda x: topo_sorted_nodes_names.index(x.name))

        for node, hessian in zip(sorted_target_nodes, trace_hessian):
            single_node_request = self._construct_single_node_request(trace_hessian_request.mode,
                                                                      trace_hessian_request.granularity,
                                                                      node)

            # The hessian for each node is expected to be a tensor where the first axis represents the number of
            # images in the batch on which the approximation was computed.
            # We collect the results as a list of a result for images, which is combined across batches.
            # After conversion, trace_hessian_request_to_score_list for a request of a single node should be a list of
            # results of all images, where each result is a tensor of the shape depending on the granularity.
            if single_node_request in self.trace_hessian_request_to_score_list:
                # self.trace_hessian_request_to_score_list[single_node_request].append(hessian)
                self.trace_hessian_request_to_score_list[single_node_request] += (
                    self._convert_tensor_to_list_of_appx_results(hessian))
            else:
                self.trace_hessian_request_to_score_list[single_node_request] = (
                    # [hessian])
                    self._convert_tensor_to_list_of_appx_results(hessian))

        # TODO: this needs to be explained carefully
        return next_iter_remain_samples if next_iter_remain_samples is not None and len(next_iter_remain_samples) > 0 \
        and len(next_iter_remain_samples[0]) > 0 else None

    def fetch_hessian(self,
                      trace_hessian_request: TraceHessianRequest,
                      required_size: int,
                      batch_size: int = 1) -> List[List[float]]:
        """
        Fetches the computed approximations of the trace of the Hessian for the given 
        request and required size.

        Args:
            trace_hessian_request: Configuration for which to fetch the approximation.
            required_size: Number of approximations required.

        Returns:
            List[List[float]]: For each target node, returnes a list of computed approximations.
            The outer list is per image (thus, has the length as required_size).
            The inner list length dependent on the granularity (1 for per-tensor, 
            OC for per-output-channel when the requested node has OC output-channels, etc.)
        """
        if required_size == 0:
            return []

        Logger.info(f"\nEnsuring {required_size} Hessian-trace approximation for nodes "
                    f"{trace_hessian_request.target_nodes}.")

        # Replace node in reused target nodes with a representing node from the 'reuse group'.
        for n in trace_hessian_request.target_nodes:
            if n.reuse_group:
                rep_node = self._get_representing_of_reuse_group(n)
                trace_hessian_request.target_nodes.remove(n)
                if rep_node not in trace_hessian_request.target_nodes:
                    trace_hessian_request.target_nodes.append(rep_node)

        # Ensure the saved info has the required number of approximations
        self._populate_saved_info_to_size(trace_hessian_request, required_size, batch_size)

        # Return the saved approximations for the given request
        return self._collect_saved_hessians_for_request(trace_hessian_request, required_size)

    def _get_representing_of_reuse_group(self, node) -> Any:
        """
        For each reused group we compute and fetch its members using a single request.
        This method creates and returns a request for the reused group the node is in.

        Args:
            node: The node to get its reuse group representative node.

        Returns: A reuse group representative node (BaseNode).
        """
        father_nodes = [n for n in self.graph.nodes if not n.reuse and n.reuse_group == node.reuse_group]
        if len(father_nodes) != 1:
            Logger.critical(f"Expected a single non-reused node in the reused group, but found {len(father_nodes)}.")

        return father_nodes[0]

    def _populate_saved_info_to_size(self,
                                     trace_hessian_request: TraceHessianRequest,
                                     required_size: int,
                                     batch_size: int = 1):
        """
        Ensures that the saved info has the required size of trace Hessian approximations for the given request.

        Args:
            trace_hessian_request: Configuration for which to ensure the saved info size.
            required_size: Required number of trace Hessian approximations.
        """

        # Get the current number of saved approximations for each node in the request
        current_existing_hessians = self.count_saved_info_of_request(trace_hessian_request)

        # Compute the required number of approximations to meet the required size.
        # Since we allow batch and multi-nodes computation, we take the node with the maximal number of missing
        # approximations to compute, and run batch computations until meeting the requirement.
        min_exist_hessians = min(current_existing_hessians.values())
        max_remaining_hessians = required_size - min_exist_hessians

        Logger.info(
            f"Running Hessian approximation computation for {len(trace_hessian_request.target_nodes)} nodes.\n "
            f"The node with minimal existing Hessian-trace approximations has {min_exist_hessians} "
            f"approximations computed.\n"
            f"{max_remaining_hessians} approximations left to compute...")

        # We restrict the requested batch size for the Hessian


        hessian_representative_dataset = partial(self._sample_batch_representative_dataset,
                                                 num_inputs=len(self.graph.input_nodes),
                                                 representative_dataset=self.representative_dataset_gen())

        next_iter_remaining_samples = None
        while max_remaining_hessians > 0:
            # If batch_size < max_remaining_hessians then we run each computation on a batch_size of images.
            # This way, we always run a computation for a single batch.
            size_to_compute = min(max_remaining_hessians, batch_size)
            next_iter_remaining_samples = (
                self.compute(trace_hessian_request, hessian_representative_dataset, size_to_compute, batch_size,
                             last_iter_remain_samples=next_iter_remaining_samples))
            max_remaining_hessians -= size_to_compute

    def _collect_saved_hessians_for_request(self, trace_hessian_request, required_size):
        collected_results = []
        for node in trace_hessian_request.target_nodes:
            single_node_request = self._construct_single_node_request(trace_hessian_request.mode,
                                                                      trace_hessian_request.granularity,
                                                                      node)

            res_for_node = self.trace_hessian_request_to_score_list.get(single_node_request)
            if res_for_node is None:
                Logger.critical(f"Couldn't find saved Hessian approximations for node {node.name}.")
            if len(res_for_node) < required_size:
                Logger.critical(f"Missing Hessian approximations for node {node.name}, requested {required_size} "
                                f"but found only {len(res_for_node)}.")

            res_for_node = res_for_node[:required_size]

            collected_results.append(res_for_node)

        return collected_results

    @staticmethod
    def _construct_single_node_request(mode, granularity, node):
        return TraceHessianRequest(mode,
                                   granularity,
                                   target_nodes=[node])

    @staticmethod
    def _convert_tensor_to_list_of_appx_results(t: Any):
        # keep the dims of the tensor
        # return [t[i:i+1, :] for i in range(t.shape[0])]
        return [t[i] for i in range(t.shape[0])]
