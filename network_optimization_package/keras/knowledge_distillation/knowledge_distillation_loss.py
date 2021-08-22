# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================
from typing import Tuple, List

import tensorflow as tf


def normalized_tensors(x: tf.Tensor) -> tf.Tensor:
    """
    Normalize a tensor by reducing its mean (the mean is computed along the batch dimension).
    Args:
        x: Tensor to normalize.

    Returns:
        The normalized tensor.
    """
    return x - tf.reduce_mean(x, axis=-1, keepdims=True)


def batch_dot_product(x: tf.Tensor,
                      y: tf.Tensor) -> tf.Tensor:
    """
    Compute the dot product of two tensors (along the batch dimension).
    Args:
        x: First tensor.
        y: Second tensor.

    Returns:
        The dot product of the tensors.
    """
    return tf.reduce_sum(x * y, axis=-1)


def cs_loss(y: tf.Tensor,
            x: tf.Tensor,
            eps: float = 1e-6) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Compute the cosine similarity of two tensors.
    Args:
        y: First tensor.
        x: Second tensor.
        eps: Small number to avoid zero division.

    Returns:
        The cosine similarity of two tensors.
    """
    y_flat = tf.reshape(y, [y.shape[0], -1])
    x_flat = tf.reshape(x, [x.shape[0], -1])
    norm_x = tf.sqrt(tf.reduce_sum(tf.pow(x_flat, 2.0), axis=-1))
    norm_y = tf.sqrt(tf.reduce_sum(tf.pow(y_flat, 2.0), axis=-1))
    cs = batch_dot_product(y_flat, x_flat) / (norm_x * norm_y + eps)
    return cs, norm_x, norm_y


def normalized_cs_loss(y: tf.Tensor,
                       x: tf.Tensor,
                       eps: float = 1e-6) -> tf.Tensor:
    """
    Compute the cosine similarity of two tensors after normalizing them.
    Args:
        y: First tensor.
        x: Second tensor.
        eps: Small number to avoid zero division.

    Returns:
        The cosine similarity of two normalized tensors.
    """
    y = tf.reshape(y, [y.shape[0], -1])
    x = tf.reshape(x, [x.shape[0], -1])
    x = normalized_tensors(x)
    y = normalized_tensors(y)
    return cs_loss(y, x, eps=eps)[0]


def multiple_tensors_cs_loss(y_list: List[tf.Tensor],
                             x_list: List[tf.Tensor],
                             weights: tf.Tensor = None) -> tf.Tensor:
    """
    Compute cosine similarity between two lists of tensors. The returned similarity
    is a list of the cosine similarity between each pair in the lists.
    If weights are passed, each element in the result is weighted by the weight at the
    index the tensors are.

    Args:
        y_list: First list of tensors.
        x_list: Second list of tensors.
        weights: Weights to scale each cosine similarity.

    Returns:
        List of cosine similarities.
    """

    loss_values_list = []
    for i, (y, x) in enumerate(zip(y_list, x_list)):
        mse = tf.reduce_mean(tf.pow(y - x, 2.0))
        ncs = tf.reduce_mean(1 - normalized_cs_loss(y, x))
        point_loss = mse + ncs
        if len(y.shape) == 4:
            mean_y = tf.reduce_mean(y, axis=[0, 1, 2], keepdims=True)
            mean_x = tf.reduce_mean(x, axis=[0, 1, 2], keepdims=True)
            mean_loss = tf.reduce_mean(tf.pow(mean_y - mean_x, 2.0))
            point_loss += mean_loss
        if weights is not None:
            point_loss *= weights[i]
        loss_values_list.append(point_loss)

    return tf.reduce_mean(tf.stack(loss_values_list))


def mve_loss(y_list: List[tf.Tensor],
             x_list: List[tf.Tensor],
             alpha: tf.Variable) -> tf.Tensor:
    """
    Compute a weighted MSE between two tensor lists.

    Args:
        y_list: First list of tensors.
        x_list: Second list of tensors.
        alpha: Variables to weight each cosine similarity.

    Returns:
        List of errors between the two lists.
    """

    loss_values_list = []
    for i, (y, x) in enumerate(zip(y_list, x_list)):
        alpha_i = alpha[i]
        mse = tf.exp(-alpha_i) * tf.reduce_mean(tf.pow(y - x, 2.0)) + alpha_i
        loss_values_list.append(mse)

    return tf.reduce_mean(tf.stack(loss_values_list))
