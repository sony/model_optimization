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

from typing import Any

import numpy as np

from model_compression_toolkit.constants import EPS
from model_compression_toolkit.logger import Logger


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


def flatten_tensor(t: np.ndarray, batch: bool, axis: int = None) -> np.ndarray:
    """
    Flattening the samples batch to allow similarity analysis computation per sample.

    Args:
        t: A tensor to be flattened.
        batch: Whether the similarity computation is per image or per tensor.
        axis: Axis along which the operator has been computed.

    Returns: A flattened tensor which has the number of samples as is first dimension.

    """

    if axis is not None and batch:
        t = np.moveaxis(t, axis, -1)
        f_t = t.reshape([t.shape[0], -1, t.shape[-1]])
    elif axis is not None:
        t = np.moveaxis(t, axis, -1)
        f_t = t.reshape([-1, t.shape[-1]])
    elif batch:
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
                batch: bool = False,
                axis: int = None,
                weights: np.ndarray = None) -> float:
    """
    Compute the mean square error between two numpy arrays.

    Args:
        float_tensor: First tensor to compare.
        fxp_tensor: Second tensor to compare.
        norm: whether to normalize the error function result.
        norm_eps: epsilon value for error normalization stability.
        batch: Whether to run batch similarity analysis or not.
        axis: Axis along which the operator has been computed.
        weights: Weights tensor to use for computing Weighted-MSE error computation.

    Returns:
        The MSE distance between the two tensors.
    """
    validate_before_compute_similarity(float_tensor, fxp_tensor)

    float_flat = flatten_tensor(float_tensor, batch, axis)
    fxp_flat = flatten_tensor(fxp_tensor, batch, axis)

    if weights is not None:
        w_flat = flatten_tensor(weights, batch, axis)
        if w_flat.shape != float_flat.shape:
            Logger.critical(f"Shape mismatch: The shape of the weights tensor {weights.shape} does not match the shape "
                            f"of the input tensors {float_flat.shape} for Weighted-MSE computation.")  # pragma: no cover
        error = ((w_flat * (float_flat - fxp_flat)) ** 2).mean(axis=-1)
    else:
        error = ((float_flat - fxp_flat) ** 2).mean(axis=-1)

    if norm:
        error /= ((float_flat ** 2).mean(axis=-1) + norm_eps)

    return error


def compute_mae(float_tensor: np.ndarray,
                fxp_tensor: np.ndarray,
                norm: bool = False,
                norm_eps: float = 1e-8,
                batch: bool = False,
                axis: int = None) -> float:
    """
    Compute the mean average error function between two numpy arrays.

    Args:
        float_tensor: First tensor to compare.
        fxp_tensor: Second tensor to compare.
        norm: whether to normalize the error function result.
        norm_eps: epsilon value for error normalization stability.
        batch: Whether to run batch similarity analysis or not.
        axis: Axis along which the operator has been computed.

    Returns:
        The mean average distance between the two tensors.
    """

    validate_before_compute_similarity(float_tensor, fxp_tensor)

    float_flat = flatten_tensor(float_tensor, batch, axis)
    fxp_flat = flatten_tensor(fxp_tensor, batch, axis)

    error = np.abs(float_flat - fxp_flat).mean(axis=-1)
    if norm:
        error /= (np.abs(float_flat).mean(axis=-1) + norm_eps)
    return error


def compute_cs(float_tensor: np.ndarray,
               fxp_tensor: np.ndarray,
               eps: float = 1e-8,
               batch: bool = False,
               axis: int = None) -> float:
    """
    Compute the similarity between two tensor using cosine similarity.
    The returned values is between 0 to 1: the smaller returned value,
    the greater similarity there is between the two tensors.

    Args:
        float_tensor: First tensor to compare.
        fxp_tensor: Second tensor to compare.
        eps: Small value to avoid zero division.
        batch: Whether to run batch similarity analysis or not.
        axis: Axis along which the operator has been computed.

    Returns:
        The cosine similarity between two tensors.
    """

    validate_before_compute_similarity(float_tensor, fxp_tensor)
    if np.all(fxp_tensor == 0) and np.all(float_tensor == 0):
        return 1.0

    float_flat = flatten_tensor(float_tensor, batch, axis)
    fxp_flat = flatten_tensor(fxp_tensor, batch, axis)

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
                    batch: bool = False,
                    axis: int = None) -> float:
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
        axis: Axis along which the operator has been computed.

    Returns:
        The Lp-norm distance between the two tensors.
    """
    validate_before_compute_similarity(float_tensor, fxp_tensor)

    float_flat = flatten_tensor(float_tensor, batch, axis)
    fxp_flat = flatten_tensor(fxp_tensor, batch, axis)

    error = (np.abs(float_flat - fxp_flat) ** p).mean(axis=-1)
    if norm:
        error /= ((np.abs(float_flat) ** p).mean(axis=-1) + norm_eps)
    return error


def compute_kl_divergence(float_tensor: np.ndarray, fxp_tensor: np.ndarray, batch: bool = False,
                          axis: int = None) -> float:
    """
    Compute the similarity between two tensor using KL-divergence.
    The returned values is between 0 and 1: the smaller returned value,
    the greater similarity there is between the two tensors.

    Args:
        float_tensor: First tensor to compare.
        fxp_tensor: Second tensor to compare.
        batch: Whether to run batch similarity analysis or not.
        axis: Axis along which the operator has been computed.

    Returns:
        The KL-divergence between two tensors.
    """

    validate_before_compute_similarity(float_tensor, fxp_tensor)

    float_flat = flatten_tensor(float_tensor, batch, axis)
    fxp_flat = flatten_tensor(fxp_tensor, batch, axis)

    non_zero_fxp_tensor = fxp_flat.copy()
    non_zero_fxp_tensor[non_zero_fxp_tensor == 0] = EPS

    prob_distance = np.where(float_flat != 0, float_flat * np.log(float_flat / non_zero_fxp_tensor), 0)
    # The sum is part of the KL-Divergence function.
    # The mean is to aggregate the distance between each output probability vectors.
    return np.mean(np.sum(prob_distance, axis=-1), axis=-1)
