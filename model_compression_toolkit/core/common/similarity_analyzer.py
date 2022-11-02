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


def tensor_batch_axis(shape: Tuple, batch: bool = False) -> Tuple:
    """
    Returns a list of axis for similarity analysis in case we want to run it on a batch
    instead of a single image/flattened tensor.

    Args:
        shape: The shape of the data tensor to run similarity analysis on.
        batch: Whether to run batch similarity analysis or not.

    Returns: If not running on batch, returns None, otherwise, returns a tuple with all axis in the tensor shape except
     the first one (batch axis).

    """

    return None if not batch else tuple([i for i in range(1, len(shape))])


def tensor_norm(x: np.ndarray, p: float = 2.0, batch: bool = False) -> np.float:
    """
    Compute the Lp-norm of a tensor x.
    Args:
        x: Tensor to compute its norm
        p: P to use for the Lp norm computation
        batch: Whether to run batch similarity analysis or not.

    Returns:
        Lp norm of x.
    """

    axis = tensor_batch_axis(x.shape, batch)
    return np.power(np.power(np.abs(x), p).sum(axis=axis), 1.0/p)


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
    axis = tensor_batch_axis(float_tensor.shape, batch)
    error = np.power(float_tensor - fxp_tensor, 2.0).mean(axis=axis)
    if norm:
        error /= (np.power(float_tensor, 2.0).mean(axis=axis) + norm_eps)
    return error


def compute_nmse(float_tensor: np.ndarray, fxp_tensor: np.ndarray) -> float:
    """
    Compute the normalized mean square error between two numpy arrays.

    Args:
        float_tensor: First tensor to compare.
        fxp_tensor: Second tensor to compare.

    Returns:
        The NMSE distance between the two tensors.
    """
    validate_before_compute_similarity(float_tensor, fxp_tensor)
    normalized_float_tensor = float_tensor / tensor_norm(float_tensor)
    normalized_fxp_tensor = fxp_tensor / tensor_norm(fxp_tensor)
    return np.mean(np.power(normalized_float_tensor - normalized_fxp_tensor, 2.0))


def compute_nmae(float_tensor: np.ndarray, fxp_tensor: np.ndarray) -> float:
    """
    Compute the normalized mean average error between two numpy arrays.

    Args:
        float_tensor: First tensor to compare.
        fxp_tensor: Second tensor to compare.

    Returns:
        The NMAE distance between the two tensors.
    """
    validate_before_compute_similarity(float_tensor, fxp_tensor)
    normalized_float_tensor = float_tensor / tensor_norm(float_tensor, 1.0)
    normalized_fxp_tensor = fxp_tensor / tensor_norm(fxp_tensor, 1.0)
    return np.mean(np.abs(normalized_float_tensor - normalized_fxp_tensor))


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
    axis = tensor_batch_axis(float_tensor.shape, batch)
    error = np.abs(float_tensor - fxp_tensor).mean(axis=axis)
    if norm:
        error /= (np.abs(float_tensor).mean(axis=axis) + norm_eps)
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

    flatten_shape = -1 if not batch else (float_tensor.shape[0], -1)
    float_flat = float_tensor.reshape(flatten_shape)
    fxp_flat = fxp_tensor.reshape(flatten_shape)
    float_norm = tensor_norm(float_flat, batch=batch)
    fxp_norm = tensor_norm(fxp_flat, batch=batch)

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
    axis = tensor_batch_axis(float_tensor.shape, batch)
    error = np.power(np.abs(float_tensor - fxp_tensor), p).mean(axis=axis)
    if norm:
        error /= (np.power(np.abs(float_tensor), p).mean(axis=axis) + norm_eps)
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
    axis = tensor_batch_axis(float_tensor.shape, batch)
    non_zero_fxp_tensor = fxp_tensor.copy()
    non_zero_fxp_tensor[non_zero_fxp_tensor == 0] = EPS
    return np.sum(np.where(float_tensor != 0, float_tensor * np.log(float_tensor / non_zero_fxp_tensor), 0), axis=axis)
