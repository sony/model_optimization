# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
import tensorflow as tf


def sample_gumbel(shape, eps=1e-6) -> tf.Tensor:
    """
    A function that sample a tensor of i.i.d gumbel random variable.
    Args:
        shape: The tensor output shape
        eps: A small number for numeric stability.

    Returns: A tensor of i.i.d gumbel random variable.

    """
    u = tf.random.uniform(shape)
    return -tf.math.log(-tf.math.log(u + eps) + eps)


def gumbel_softmax(in_pi: tf.Tensor, in_tau: tf.Tensor, in_gumbel: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
    """
    A gumbel softmax function.
    Args:
        in_pi: A tensor of log probability.
        in_tau: A temperature tensor.
        in_gumbel: A tensor of gumbel random variable.
        eps: A small number for numeric stability.

    Returns: A gumbel softmax probability tensor.

    """
    return tf.nn.softmax((in_pi + in_tau * in_gumbel) / (in_tau + eps), axis=0)


def ste_gumbel(in_prob: tf.Tensor) -> tf.Tensor:
    """
    This function apply ste on the output of the gumbel softmax.
    Args:
        in_prob:A tensor of probability

    Returns: A Tensor of ohe hot vector with STE.

    """

    delta = tf.stop_gradient(select_gumbel(in_prob) - in_prob)
    return in_prob + delta


def select_gumbel(in_prob: tf.Tensor) -> tf.Tensor:
    """
    This function apply ste on the output of the gumbel softmax.
    Args:
        in_prob: A tensor of probability.

    Returns: A Tensor of ohe hot vector

    """
    max_index = tf.argmax(in_prob, axis=0)
    one_hot_prob = tf.one_hot(max_index, depth=in_prob.shape[0], axis=0)
    return one_hot_prob + 0 * in_prob
