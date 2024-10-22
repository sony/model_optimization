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
from dataclasses import dataclass
from typing import List, Dict, Tuple, TYPE_CHECKING

import numpy as np

from model_compression_toolkit.constants import HESSIAN_NUM_ITERATIONS
from model_compression_toolkit.core.common.hessian.hessian_scores_request import HessianScoresRequest, HessianMode, \
    HessianScoresGranularity

if TYPE_CHECKING:    # pragma: no cover
    from model_compression_toolkit.core.common import BaseNode


# type hints aliases
LayerName = str
Tensor = np.ndarray


@dataclass(eq=True, frozen=True)
class Query:
    """ Query key for hessians cache. """
    mode: HessianMode
    granularity: HessianScoresGranularity
    node: LayerName


class HessianCache:
    """ Hessian cache """
    def __init__(self):
        self._data: Dict[Query, Tensor] = {}

    def update(self, layers_hessians: Dict[str, np.ndarray], request: HessianScoresRequest) -> int:
        """
        Updates the cache with new hessians estimations.

        Note: we assume that the new hessians were computed on different samples than previously stored hessians.
        If same samples were used more than once, duplicates will be stored. This can only be a problem if hessians
        for the same query were computed via multiple requests and dataloader in each request yields same samples.
        We cannot just filter out duplicates since in some cases we can get valid identical hessians on different
        samples.

        Args:
            layers_hessians: a dictionary from layer names to their hessian score tensors.
            request: request per which hessians were computed.

        Returns:
            Minimal samples count after update (among updated layers).

        """
        assert set(layers_hessians.keys()) == set(n.name for n in request.target_nodes)
        n_nodes_samples = []   # samples count per node after update
        for node_name, hess in layers_hessians.items():
            query = Query(request.mode, request.granularity, node_name)
            saved_hess = self._data.get(query)
            new_hess = hess if saved_hess is None else np.concatenate([saved_hess, hess], axis=0)
            self._data[query] = new_hess
            n_nodes_samples.append(new_hess.shape[0])

        return min(n_nodes_samples)

    def fetch_hessian(self, request: HessianScoresRequest) -> Tuple[Dict[LayerName, Tensor], Dict[LayerName, int]]:
        """
        Fetch available hessians per request and identify missing samples.

        Note: if fewer samples are available than requested, hessians tensor will contain the available samples.

        Args:
            request: hessians fetch request.

        Returns:
            A tuple of two dictionaries:
            - A dictionary from layer name to a tensor of its hessian.
            - A dictionary from layer name to a number of missing samples.
        """
        assert request.n_samples is not None

        result = {}
        missing = {}
        for node in request.target_nodes:
            query = Query(request.mode, request.granularity, node.name)
            hess = self._data.get(query)
            if hess is None:
                missing[node.name] = request.n_samples
                continue
            n_missing = request.n_samples - hess.shape[0]
            if n_missing > 0:
                missing[node.name] = n_missing
            result[node.name] = hess[:request.n_samples, ...]

        return result, missing

    def clear(self):
        """ Clear the cache. """
        self._data.clear()


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
                 fw_impl,
                 num_iterations_for_approximation: int = HESSIAN_NUM_ITERATIONS):
        """
        Args:
            graph: Float graph.
            fw_impl: Framework-specific implementation for Hessian approximation scores computation.
            num_iterations_for_approximation: the number of iterations for hessian estimation.
        """
        self.graph = graph
        self.fw_impl = fw_impl
        self.num_iterations_for_approximation = num_iterations_for_approximation
        self.cache = HessianCache()

    def fetch_hessian(self, request: HessianScoresRequest,
                      force_compute: bool = False) -> Dict[LayerName, Tensor]:
        """
        Fetch hessians per request.
        If 'force_compute' is False, will first try to retrieve previously cached hessians. If no or not enough
        hessians are found in the cache, will compute the remaining number of hessians to fulfill the request.
        If 'force_compute' is True, will compute the hessians (use when you need hessians for specific inputs).

        Args:
            request: request per which to fetch the hessians.
            force_compute: if True, will compute the hessians.
                           If False, will look for cached hessians first.

        Returns:
            A dictionary of layers' hessian tensors of shape (samples, ...). The exact shape depends on the
            requested granularity.
        """
        if request.n_samples is None and not force_compute:
            raise ValueError('Number of samples can be None only when force_compute is True.')

        orig_request = request
        # replace reused nodes with primary nodes
        # TODO need to check if there is a bug in reuse. While this is the same layer, the compare tensors and their
        #  gradients are not. It seems that currently the same compare tensor of the primary node is used multiple times
        target_nodes = [self._get_primary_node(n) for n in request.target_nodes]
        request = request.clone(target_nodes=target_nodes)

        if force_compute:
            res = self._compute_hessians(request, self.num_iterations_for_approximation, count_by_cache=False)
        else:
            res = self._fetch_hessians_with_compute(request, self.num_iterations_for_approximation)

        # restore nodes from the original request
        res = {n_orig.name: res[n.name] for n_orig, n in zip(orig_request.target_nodes, request.target_nodes)}
        return res

    def clear_cache(self):
        """ Purge the cached hessians. """
        self.cache.clear()

    def _fetch_hessians_with_compute(self, request: HessianScoresRequest, n_iterations: int) -> Dict[LayerName, Tensor]:
        """
        Fetch pre-computed hessians for the request if available. Otherwise, compute the missing hessians.

        Args:
            request: hessian estimation request.
            n_iterations: the number of iterations for hessian estimation.

        Returns:
            A dictionary from layers (by name) to their hessians.
        """
        res, missing = self.cache.fetch_hessian(request)
        if not missing:
            return res

        if request.data_loader is None:
            raise ValueError(f'Not enough hessians are cached to fulfill the request, but data loader was not passed '
                             f'for additional computation. Requested {request.n_samples}, '
                             f'available {min(missing.values())}.')

        orig_request = request
        # if some hessians were found generate a new request only for missing nodes.
        if res:
            target_nodes = [n for n in orig_request.target_nodes if n.name in missing]
            request = request.clone(target_nodes=target_nodes)
        self._compute_hessians(request, n_iterations, count_by_cache=True)
        res, missing = self.cache.fetch_hessian(request)
        assert not missing
        return res

    def _compute_hessians(self, request: HessianScoresRequest,
                          n_iterations: int, count_by_cache: bool) -> Dict[LayerName, Tensor]:
        """
        Computes hessian estimation per request.

        Data loader from request is used as is, i.e. it should reflect the required batch size (e.g. if
        hessians should be estimated sample by sample, the data loader should yield a single sample at a time).

        NOTE: the returned value only contains hessians that were computed here, which may differ from the requested
          number of samples. It's only intended for use when you specifically need sample-wise hessians for the
          samples in the request.

        Args:
            request: hessian estimation request.
            n_iterations: the number of iterations for hessian estimation.
            count_by_cache: if True, computes hessians until the cache contains the requested number of samples.
                            if False, computes hessian for the first requested number of sample in the dataloader.
        Returns:
            A dictionary from layers (by name) to their hessian tensors that *were computed in this invocation*.
            First axis corresponds to samples in the order determined by the data loader.
        """
        if count_by_cache:
            assert request.n_samples is not None

        n_samples = 0
        hess_per_layer = []
        for batch in request.data_loader:
            batch_hess_per_layer = self._compute_hessian_for_batch(request, batch, n_iterations)
            hess_per_layer.append(batch_hess_per_layer)
            min_count = self.cache.update(batch_hess_per_layer, request)
            n_samples = min_count if count_by_cache else (n_samples + batch[0].shape[0])
            if request.n_samples and n_samples >= request.n_samples:
                break

        hess_per_layer = {
            layer.name: np.concatenate([hess[layer.name] for hess in hess_per_layer], axis=0)
            for layer in request.target_nodes
        }

        if request.n_samples:
            if n_samples < request.n_samples:
                raise ValueError(f'Could not compute the requested number of Hessians ({request.n_samples}), '
                                 f'not enough samples in the provided representative dataset.')

            if n_samples > request.n_samples:
                hess_per_layer = {
                    layer: hess[:request.n_samples, ...] for layer, hess in hess_per_layer.items()
                }
        return hess_per_layer

    def _compute_hessian_for_batch(self,
                                   request: HessianScoresRequest,
                                   inputs_batch: List[Tensor],
                                   n_iterations: int) -> Dict[LayerName, Tensor]:
        """
        Use hessian score calculator to compute hessian approximations for a batch of inputs.

        Args:
            request: hessian estimation request.
            inputs_batch: a batch of inputs to estimate hessians on.
            n_iterations: the number of iterations for hessian estimation.

        Returns:
            A dictionary from layers (by name) to their hessians.
        """
        fw_hessian_calculator = self.fw_impl.get_hessian_scores_calculator(
            graph=self.graph,
            input_images=inputs_batch,
            hessian_scores_request=request,
            num_iterations_for_approximation=n_iterations
        )

        hessian_scores: list = fw_hessian_calculator.compute()

        layers_hessian_scores = {
            layer.name: score for layer, score in zip(request.target_nodes, hessian_scores)
        }
        return layers_hessian_scores

    def _get_primary_node(self, node: 'BaseNode') -> 'BaseNode':
        """
        Get node's primary node that it reuses, or itself if not reused.

        Args:
            node: node's object to get its primary node.

        Returns:
            Node's primary node.
        """
        if node.reuse is False:
            return node

        father_nodes = [n for n in self.graph.nodes if not n.reuse and n.reuse_group == node.reuse_group]
        assert len(father_nodes) == 1
        return father_nodes[0]
