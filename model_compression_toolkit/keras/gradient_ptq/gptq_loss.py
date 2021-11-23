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

from typing import Any, Tuple, List

import tensorflow as tf


def mse_loss(y: tf.Tensor, x: tf.Tensor, normalized: bool = True) -> tf.Tensor:
    """
    Compute the MSE of two tensors.
    Args:
        y: First tensor.
        x: Second tensor.
        normalized: either return normalized MSE (default), or MSE

    Returns:
        The MSE of two tensors.
    """
    _loss = tf.reduce_mean(tf.pow(y - x, 2.0))
    return _loss if normalized else _loss / tf.reduce_mean(tf.pow(x, 2.0))


def multiple_tensors_mse_loss(y_list: List[tf.Tensor],
                              x_list: List[tf.Tensor]) -> tf.Tensor:
    """
    Compute MSE similarity between two lists of tensors

    Args:
        y_list: First list of tensors.
        x_list: Second list of tensors.

    Returns:
        A single loss value which is the average of all MSE loss of all tensor pairs
        List of MSE similarities per tensor pair
    """

    loss_values_list = []
    for i, (y, x) in enumerate(zip(y_list, x_list)):
        point_loss = mse_loss(y, x)
        loss_values_list.append(point_loss)

    return tf.reduce_mean(tf.stack(loss_values_list))
