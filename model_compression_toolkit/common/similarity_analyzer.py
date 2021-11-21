# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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

#########################
#  Helpful functions
#########################


def validate_before_compute_similarity(a: Any, b: Any):
    """
    Assert different conditions before comparing two tensors (such as dimensionality and type).
    Args:
        a: First tensor to validate.
        b: Second tensor to validate.

    """
    assert isinstance(a, np.ndarray)
    assert isinstance(b, np.ndarray)
    assert a.shape == b.shape


def tensor_norm(x: np.ndarray) -> np.float:
    """
    Compute the L2-norm of a tensor x.
    Args:
        x: Tensor to compute its norm

    Returns:
        L2 norm of x.
    """
    return np.sqrt(np.power(x.flatten(), 2.0).sum())


#########################
# Similarity functions
#########################


def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the mean square error between two numpy arrays.

    Args:
        a: First tensor to compare.
        b: Second tensor to compare.

    Returns:
        The MSE distance between the two tensors.
    """
    validate_before_compute_similarity(a, b)
    return np.power(a - b, 2.0).mean()


def compute_nmse(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the normalized mean square error between two numpy arrays.

    Args:
        a: First tensor to compare.
        b: Second tensor to compare.

    Returns:
        The NMSE distance between the two tensors.
    """
    validate_before_compute_similarity(a, b)
    return np.power(a - b, 2.0).mean() / np.power(tensor_norm(a - a.mean()), 2.0)


def compute_mae(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the mean average error function between two numpy arrays.

    Args:
        a: First tensor to compare.
        b: Second tensor to compare.

    Returns:
        The mean average distance between the two tensors.
    """

    validate_before_compute_similarity(a, b)
    return np.abs(a - b).mean()


def compute_cs(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute the similarity between two tensor using cosine similarity.
    The returned values is between 0 to 1: the smaller returned value,
    the greater similarity there is between the two tensors.

    Args:
        a: First tensor to compare.
        b: Second tensor to compare.
        eps: Small value to avoid zero division.

    Returns:
        The cosine similarity between two tensors.
    """

    validate_before_compute_similarity(a, b)
    if np.all(b == 0) and np.all(a == 0):
        return 1.0

    a_flat = a.flatten()
    b_flat = b.flatten()
    a_norm = tensor_norm(a)
    b_norm = tensor_norm(b)

    # -1 <= cs <= 1
    cs = np.sum(a_flat * b_flat) / ((a_norm * b_norm) + eps)

    # Return a non-negative float (smaller value -> more similarity)
    return (1.0-cs)/2.0


def compute_lp_norm(a: np.ndarray, b: np.ndarray, p: int) -> float:
    """
    Compute the error function between two numpy arrays.
    The error is computed based on Lp-norm distance of the tensors.

    Args:
        a: First tensor to compare.
        b: Second tensor to compare.
        p: p-norm to use for the Lp-norm distance.

    Returns:
        The Lp-norm distance between the two tensors.
    """
    validate_before_compute_similarity(a, b)
    return np.power(np.abs(a - b), p).mean()


