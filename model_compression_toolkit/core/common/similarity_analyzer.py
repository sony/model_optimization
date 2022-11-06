# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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

from typing import Any, Tuple

import numpy as np

from model_compression_toolkit.core.common.constants import EPS

#########################
#  Helpful functions
#########################


def validate_before_compute_similarity(float_tensor: Any, fxp_tensor: Any):
    """
    Assert different conditions before comparing two tensors (such as dimensionality and type).
    Args:
        float_tensor: First tensor to validate.
        fxp_tensor: Second tensor to validate.

    """
    assert isinstance(float_tensor, np.ndarray)
    assert isinstance(fxp_tensor, np.ndarray)
    assert float_tensor.shape == fxp_tensor.shape


def _similarity_tensor_norm(x: np.ndarray, p: float = 2.0) -> np.ndarray:
    """
    Compute the Lp-norm of a tensor x.
    Args:
        x: Tensor to compute its norm
        p: P to use for the Lp norm computation

    Returns:
        Lp norm per sample in batch of tensor x.
    """

    return (np.abs(x) ** p).sum(axis=-1) ** (1.0/p)


def flatten_tensor(t: np.ndarray, batch: bool) -> np.ndarray:
    """
    Flattening the samples batch to allow similarity analysis computation per sample.

    Args:
        t: A tensor to be flattened.
        batch: Whether the similarity computation is per image or per tensor.

    Returns: A flattened tensor which has the number of samples as is first dimension.

    """

    if batch:
        f_t = t.reshape([t.shape[0], -1])
    else:
        f_t = t.flatten()

    return f_t

#########################
# Similarity functions
#########################


def compute_mse(float_tensor: np.ndarray,
                fxp_tensor: np.ndarray,
                norm: bool = False,
                norm_eps: float = 1e-8,
                batch: bool = False) -> float:
    """
    Compute the mean square error between two numpy arrays.

    Args:
        float_tensor: First tensor to compare.
        fxp_tensor: Second tensor to compare.
        norm: whether to normalize the error function result.
        norm_eps: epsilon value for error normalization stability.
        batch: Whether to run batch similarity analysis or not.

    Returns:
        The MSE distance between the two tensors.
    """
    validate_before_compute_similarity(float_tensor, fxp_tensor)

    float_flat = flatten_tensor(float_tensor, batch)
    fxp_flat = flatten_tensor(fxp_tensor, batch)

    error = ((float_flat - fxp_flat) ** 2).mean(axis=-1)
    if norm:
        error /= ((float_flat ** 2).mean(axis=-1) + norm_eps)

    return error


def compute_mae(float_tensor: np.ndarray,
                fxp_tensor: np.ndarray,
                norm: bool = False,
                norm_eps: float = 1e-8,
                batch: bool = False) -> float:
    """
    Compute the mean average error function between two numpy arrays.

    Args:
        float_tensor: First tensor to compare.
        fxp_tensor: Second tensor to compare.
        norm: whether to normalize the error function result.
        norm_eps: epsilon value for error normalization stability.
        batch: Whether to run batch similarity analysis or not.

    Returns:
        The mean average distance between the two tensors.
    """

    validate_before_compute_similarity(float_tensor, fxp_tensor)

    float_flat = flatten_tensor(float_tensor, batch)
    fxp_flat = flatten_tensor(fxp_tensor, batch)

    error = np.abs(float_flat - fxp_flat).mean(axis=-1)
    if norm:
        error /= (np.abs(float_flat).mean(axis=-1) + norm_eps)
    return error


def compute_cs(float_tensor: np.ndarray, fxp_tensor: np.ndarray, eps: float = 1e-8, batch: bool = False) -> float:
    """
    Compute the similarity between two tensor using cosine similarity.
    The returned values is between 0 to 1: the smaller returned value,
    the greater similarity there is between the two tensors.

    Args:
        float_tensor: First tensor to compare.
        fxp_tensor: Second tensor to compare.
        eps: Small value to avoid zero division.
        batch: Whether to run batch similarity analysis or not.

    Returns:
        The cosine similarity between two tensors.
    """

    validate_before_compute_similarity(float_tensor, fxp_tensor)
    if np.all(fxp_tensor == 0) and np.all(float_tensor == 0):
        return 1.0

    float_flat = flatten_tensor(float_tensor, batch)
    fxp_flat = flatten_tensor(fxp_tensor, batch)

    float_norm = _similarity_tensor_norm(float_flat)
    fxp_norm = _similarity_tensor_norm(fxp_flat)

    # -1 <= cs <= 1
    axis = None if not batch else 1
    cs = np.sum(float_flat * fxp_flat, axis=axis) / ((float_norm * fxp_norm) + eps)

    # Return a non-negative float (smaller value -> more similarity)
    return (1.0 - cs) / 2.0


def compute_lp_norm(float_tensor: np.ndarray,
                    fxp_tensor: np.ndarray,
                    p: int,
                    norm: bool = False,
                    norm_eps: float = 1e-8,
                    batch: bool = False) -> float:
    """
    Compute the error function between two numpy arrays.
    The error is computed based on Lp-norm distance of the tensors.

    Args:
        float_tensor: First tensor to compare.
        fxp_tensor: Second tensor to compare.
        p: p-norm to use for the Lp-norm distance.
        norm: whether to normalize the error function result.
        norm_eps: epsilon value for error normalization stability.
        batch: Whether to run batch similarity analysis or not.

    Returns:
        The Lp-norm distance between the two tensors.
    """
    validate_before_compute_similarity(float_tensor, fxp_tensor)

    float_flat = flatten_tensor(float_tensor, batch)
    fxp_flat = flatten_tensor(fxp_tensor, batch)

    error = (np.abs(float_flat - fxp_flat) ** p).mean(axis=-1)
    if norm:
        error /= ((np.abs(float_flat) ** p).mean(axis=-1) + norm_eps)
    return error


def compute_kl_divergence(float_tensor: np.ndarray, fxp_tensor: np.ndarray, batch: bool = False) -> float:
    """
    Compute the similarity between two tensor using KL-divergence.
    The returned values is between 0 to 1: the smaller returned value,
    the greater similarity there is between the two tensors.

    Args:
        float_tensor: First tensor to compare.
        fxp_tensor: Second tensor to compare.
        batch: Whether to run batch similarity analysis or not.

    Returns:
        The KL-divergence between two tensors.
    """

    validate_before_compute_similarity(float_tensor, fxp_tensor)

    float_flat = flatten_tensor(float_tensor, batch)
    fxp_flat = flatten_tensor(fxp_tensor, batch)

    non_zero_fxp_tensor = fxp_flat.copy()
    non_zero_fxp_tensor[non_zero_fxp_tensor == 0] = EPS

    return np.sum(np.where(float_flat != 0, float_flat * np.log(float_flat / non_zero_fxp_tensor), 0), axis=-1)
