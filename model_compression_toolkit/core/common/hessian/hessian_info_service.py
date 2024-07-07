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

import numpy as np
from functools import partial
from tqdm import tqdm
from typing import Callable, List, Dict, Any, Tuple

from model_compression_toolkit.constants import HESSIAN_NUM_ITERATIONS
from model_compression_toolkit.core.common.hessian.hessian_scores_request import HessianScoresRequest, \
    HessianScoresGranularity, HessianMode
from model_compression_toolkit.logger import Logger


class HessianInfoService:
    """
    A service to manage, store, and compute information based on the Hessian matrix approximation.

    This class provides functionalities to compute information based on the Hessian matrix approximation
    based on the different parameters (such as number of iterations for approximating the scores)
    and input images (using representative_dataset_gen).
    It also offers cache management capabilities for efficient computation and retrieval.

    Note:
    - The Hessian provides valuable information about the curvature of the loss function.
    - Computation can be computationally heavy and time-consuming.
    - The computed information is based on Hessian approximation (and not the precise Hessian matrix).
    """

    def __init__(self,
                 graph,
                 representative_dataset_gen: Callable,
                 fw_impl,
                 num_iterations_for_approximation: int = HESSIAN_NUM_ITERATIONS):
        """

        Args:
            graph: Float graph.
            representative_dataset_gen: A callable that provides a dataset for sampling.
            fw_impl: Framework-specific implementation for Hessian approximation scores computation.
        """
        self.graph = graph

        self.representative_dataset_gen = representative_dataset_gen

        self.fw_impl = fw_impl
        self.num_iterations_for_approximation = num_iterations_for_approximation

        self.hessian_scores_request_to_scores_list = {}

    def _sample_batch_representative_dataset(self,
                                             representative_dataset: Any,
                                             num_hessian_samples: int,
                                             num_inputs: int,
                                             last_iter_remain_samples: List[List[np.ndarray]] = None
                                             ) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
        """
        Get a batch of samples from a representative dataset with the requested num_hessian_samples.

        Args:
            representative_dataset: A generator which yields batches of input samples.
            num_hessian_samples: Number of requested samples to compute batch Hessian approximation scores.
            num_inputs: Number of input layers of the model on which the scores are computed.
            last_iter_remain_samples: A list of input samples (for each input layer) with remaining samples from
            previous iterations.

        Returns: A tuple with two lists:
            (1) A list of inputs - a tensor of the requested batch size for each input layer.
            (2) A list of remaining samples - for each input layer.
        """

        if num_inputs < 0:  # pragma: no cover
            Logger.critical(f"Number of images to compute Hessian approximation must be positive, "
                            f"but given {num_inputs}.")

        all_inp_hessian_samples = [[] for _ in range(num_inputs)]
        all_inp_remaining_samples = [[] for _ in range(num_inputs)]

        # Collect the requested number of samples from the representative dataset
        # In case there are samples left from previous iterations, we use them first
        # otherwise, we take a batch from the representative dataset generator
        while len(all_inp_hessian_samples[0]) < num_hessian_samples:
            batch = None
            sampling_from_repr = True
            if last_iter_remain_samples is not None and len(last_iter_remain_samples[0]) >= num_hessian_samples:
                batch = last_iter_remain_samples
                sampling_from_repr = False
            else:
                try:
                    batch = next(representative_dataset)
                except StopIteration:
                    Logger.critical(
                        f"Not enough samples in the provided representative dataset to compute Hessian approximation on "
                        f"{num_hessian_samples} samples.")

            if batch is not None and not isinstance(batch, list):
                Logger.critical(f'Expected batch to be a list; found type: {type(batch)}.')  # pragma: no cover

            for inp_idx in range(len(batch)):
                inp_batch = batch[inp_idx] if sampling_from_repr else np.stack(batch[inp_idx], axis=0)
                if not sampling_from_repr:
                    last_iter_remain_samples[inp_idx] = []

                # Compute number of missing samples to get to the requested amount from the current batch
                num_missing = min(num_hessian_samples - len(all_inp_hessian_samples[inp_idx]), inp_batch.shape[0])

                # Append each sample separately
                samples = [s for s in inp_batch[0:num_missing, ...]]
                remaining_samples = [s for s in inp_batch[num_missing:, ...]]

                all_inp_hessian_samples[inp_idx] += [sample.reshape(1, *sample.shape) for sample in samples]

                # This list can only get filled on the last batch iteration
                all_inp_remaining_samples[inp_idx] += remaining_samples

            if len(all_inp_hessian_samples[0]) > num_hessian_samples:
                Logger.critical(f"Requested {num_hessian_samples} samples for computing Hessian approximation but "
                                f"{len(all_inp_hessian_samples[0])} were collected.")  # pragma: no cover

        # Collected enough samples, constructing a dataset with the requested batch size
        hessian_samples_for_input = []
        for inp_samples in all_inp_hessian_samples:
            inp_samples = np.concatenate(inp_samples, axis=0)
            num_collected_samples = inp_samples.shape[0]
            inp_samples = np.split(inp_samples,
                                   num_collected_samples // min(num_collected_samples, num_hessian_samples))
            hessian_samples_for_input.append(inp_samples[0])

        return hessian_samples_for_input, all_inp_remaining_samples

    def _clear_saved_hessian_info(self):
        """Clears the saved info approximations."""
        self.hessian_scores_request_to_scores_list={}

    def count_saved_scores_of_request(self, hessian_request: HessianScoresRequest) -> Dict:
        """
        Counts the saved approximations of Hessian scores for a specific request.
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
            per_node_counter[n] = len(self.hessian_scores_request_to_scores_list.get(hessian_request, []))

        return per_node_counter

    def compute(self,
                hessian_scores_request: HessianScoresRequest,
                representative_dataset_gen,
                num_hessian_samples: int,
                last_iter_remain_samples: List[List[np.ndarray]] = None):
        """
        Computes scores based on the Hessian matrix approximation according to the
        provided request configuration and stores it in the cache.

        Args:
            hessian_scores_request: Configuration for which to compute the approximation.
            representative_dataset_gen: A callable that provides a dataset for sampling.
            num_hessian_samples: Number of requested samples to compute batch Hessian approximation scores.
            last_iter_remain_samples: A list of input samples (for each input layer) with remaining samples from
            previous iterations.
        """
        Logger.debug(f"Computing Hessian-scores approximations for nodes {hessian_scores_request.target_nodes}.")

        images, next_iter_remain_samples = representative_dataset_gen(num_hessian_samples=num_hessian_samples,
                                                                      last_iter_remain_samples=last_iter_remain_samples)

        # Compute and store the computed approximation in the saved info
        topo_sorted_nodes_names = [x.name for x in self.graph.get_topo_sorted_nodes()]
        hessian_scores_request.target_nodes.sort(key=lambda x: topo_sorted_nodes_names.index(x.name))

        # Get the framework-specific calculator Hessian-approximation scores
        fw_hessian_calculator = self.fw_impl.get_hessian_scores_calculator(graph=self.graph,
                                                                           input_images=images,
                                                                           hessian_scores_request=hessian_scores_request,
                                                                           num_iterations_for_approximation=self.num_iterations_for_approximation)

        hessian_scores = fw_hessian_calculator.compute()

        for node, hessian in zip(hessian_scores_request.target_nodes, hessian_scores):
            single_node_request = self._construct_single_node_request(hessian_scores_request.mode,
                                                                      hessian_scores_request.granularity,
                                                                      node)

            # The hessian for each node is expected to be a tensor where the first axis represents the number of
            # images in the batch on which the approximation was computed.
            # We collect the results as a list of a result for images, which is combined across batches.
            # After conversion, hessian_scores_request_to_scores_list for a request of a single node should be a list of
            # results of all images, where each result is a tensor of the shape depending on the granularity.
            if single_node_request in self.hessian_scores_request_to_scores_list:
                self.hessian_scores_request_to_scores_list[single_node_request] += (
                    self._convert_tensor_to_list_of_appx_results(hessian))
            else:
                self.hessian_scores_request_to_scores_list[single_node_request] = (
                    self._convert_tensor_to_list_of_appx_results(hessian))

        # In case that we are required to return a number of scores that is larger that the computation batch size
        # and if in this case the computation batch size is smaller than the representative dataset batch size
        # we need to carry over remaining samples from the last fetched batch to the next computation, otherwise,
        # we might skip samples or remain without enough samples to complete the computations for the
        # requested number of scores.
        return next_iter_remain_samples if next_iter_remain_samples is not None and len(next_iter_remain_samples) > 0 \
        and len(next_iter_remain_samples[0]) > 0 else None

    def fetch_hessian(self,
                      hessian_scores_request: HessianScoresRequest,
                      required_size: int,
                      batch_size: int = 1) -> List[List[np.ndarray]]:
        """
        Fetches the computed approximations of the Hessian-based scores for the given
        request and required size.

        Args:
            hessian_scores_request: Configuration for which to fetch the approximation.
            required_size: Number of approximations required.
            batch_size: The Hessian computation batch size.

        Returns:
            List[List[np.ndarray]]: For each target node, returns a list of computed approximations.
            The outer list is per image (thus, has the length as required_size).
            The inner list length dependent on the granularity (1 for per-tensor, 
            OC for per-output-channel when the requested node has OC output-channels, etc.)
        """

        if len(hessian_scores_request.target_nodes) == 0:
            return []

        if required_size == 0:
            return [[] for _ in hessian_scores_request.target_nodes]

        Logger.info(f"\nEnsuring {required_size} Hessian-approximation scores for nodes "
                    f"{hessian_scores_request.target_nodes}.")

        # Replace node in reused target nodes with a representing node from the 'reuse group'.
        for n in hessian_scores_request.target_nodes:
            if n.reuse_group:
                rep_node = self._get_representing_of_reuse_group(n)
                hessian_scores_request.target_nodes.remove(n)
                if rep_node not in hessian_scores_request.target_nodes:
                    hessian_scores_request.target_nodes.append(rep_node)

        # Ensure the saved info has the required number of approximations
        self._populate_saved_info_to_size(hessian_scores_request, required_size, batch_size)

        # Return the saved approximations for the given request
        return self._collect_saved_hessians_for_request(hessian_scores_request, required_size)

    def _get_representing_of_reuse_group(self, node) -> Any:
        """
        For each reused group we compute and fetch its members using a single request.
        This method creates and returns a request for the reused group the node is in.

        Args:
            node: The node to get its reuse group representative node.

        Returns: A reuse group representative node (BaseNode).
        """
        father_nodes = [n for n in self.graph.nodes if not n.reuse and n.reuse_group == node.reuse_group]
        if len(father_nodes) != 1:  # pragma: no cover
            Logger.critical(f"Expected a single non-reused node in the reused group, "
                            f"but found {len(father_nodes)}.")

        return father_nodes[0]

    def _populate_saved_info_to_size(self,
                                     hessian_scores_request: HessianScoresRequest,
                                     required_size: int,
                                     batch_size: int = 1):
        """
        Ensures that the saved info has the required size of Hessian approximation scores for the given request.

        Args:
            hessian_scores_request: Configuration of the request to ensure the saved info size.
            required_size: Required number of Hessian-approximation scores.
            batch_size: The Hessian computation batch size.
        """

        # Get the current number of saved approximations for each node in the request
        current_existing_hessians = self.count_saved_scores_of_request(hessian_scores_request)

        # Compute the required number of approximations to meet the required size.
        # Since we allow batch and multi-nodes computation, we take the node with the maximal number of missing
        # approximations to compute, and run batch computations until meeting the requirement.
        min_exist_hessians = min(current_existing_hessians.values())
        max_remaining_hessians = required_size - min_exist_hessians

        Logger.info(
            f"Running Hessian approximation computation for {len(hessian_scores_request.target_nodes)} nodes.\n "
            f"The node with minimal existing Hessian-approximation scores has {min_exist_hessians} "
            f"approximated scores computed.\n"
            f"{max_remaining_hessians} approximations left to compute...")

        hessian_representative_dataset = partial(self._sample_batch_representative_dataset,
                                                 num_inputs=len(self.graph.input_nodes),
                                                 representative_dataset=self.representative_dataset_gen())

        next_iter_remaining_samples = None
        pbar = tqdm(desc="Computing Hessian approximations...", total=None)
        while max_remaining_hessians > 0:
            # If batch_size < max_remaining_hessians then we run each computation on a batch_size of images.
            # This way, we always run a computation for a single batch.
            pbar.update(1)
            size_to_compute = min(max_remaining_hessians, batch_size)
            next_iter_remaining_samples = (
                self.compute(hessian_scores_request, hessian_representative_dataset, size_to_compute,
                             last_iter_remain_samples=next_iter_remaining_samples))
            max_remaining_hessians -= size_to_compute

    def _collect_saved_hessians_for_request(self,
                                            hessian_scores_request: HessianScoresRequest,
                                            required_size: int) -> List[List[np.ndarray]]:
        """
        Collects Hessian approximation for the nodes in the given request.

        Args:
            hessian_scores_request: Configuration for which to fetch the approximation.
            required_size: Required number of Hessian-approximation scores.

        Returns: A list with List of computed Hessian approximation (a tensor for each score) for each node
        in the request.

        """
        collected_results = []
        for node in hessian_scores_request.target_nodes:
            single_node_request = self._construct_single_node_request(hessian_scores_request.mode,
                                                                      hessian_scores_request.granularity,
                                                                      node)

            res_for_node = self.hessian_scores_request_to_scores_list.get(single_node_request)
            if res_for_node is None:  # pragma: no cover
                Logger.critical(f"Couldn't find saved Hessian approximations for node {node.name}.")
            if len(res_for_node) < required_size:  # pragma: no cover
                Logger.critical(f"Missing Hessian approximations for node {node.name}, requested {required_size} "
                                f"but found only {len(res_for_node)}.")

            res_for_node = res_for_node[:required_size]

            collected_results.append(res_for_node)

        return collected_results

    @staticmethod
    def _construct_single_node_request(mode: HessianMode,
                                       granularity: HessianScoresGranularity,
                                       target_nodes: List) -> HessianScoresRequest:
        """
        Constructs a Hessian request with for a single node. Used for retrieving and maintaining cached results.

        Args:
            mode (HessianMode): Mode of Hessian's approximation (w.r.t weights or activations).
            granularity (HessianScoresGranularity): Granularity level for the approximation.
            target_nodes (List[BaseNode]): The node in the float graph for which the Hessian's approximation scores is targeted.

        Returns: A HessianScoresRequest with the given details for the requested node.

        """
        return HessianScoresRequest(mode,
                                    granularity,
                                    target_nodes=[target_nodes])

    @staticmethod
    def _convert_tensor_to_list_of_appx_results(t: Any) -> List:
        """
        Converts a tensor with batch computation results to a list of individual result for each sample in batch.

        Args:
            t: A tensor with Hessian approximation results.

        Returns: A list with split batch into individual results.

        """
        return [t[i:i+1, :] for i in range(t.shape[0])]
