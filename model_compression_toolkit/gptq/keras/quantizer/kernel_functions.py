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
from model_compression_toolkit.core.keras.constants import KERNEL


def get_kernel(weights_list: list) -> tf.Tensor:
    """
    This function a list of weights and return the kernel
    Args:
        weights_list:  A list of Tensors

    Returns: The kernel tensor.

    """
    for w in weights_list:
        if KERNEL in w.name:
            return w
    raise Exception("Can't find kernel variable")


def threshold_reshape(threshold_tensor: tf.Tensor, input_w: tf.Tensor, in_quantization_axis: int) -> tf.Tensor:
    """
    This function take a threshold tensor and re-aline it to the weight tensor channel axis.
    Args:
        threshold_tensor: A tensor of threshold
        input_w: A weight tensor
        in_quantization_axis: A int value that represent the channel axis.

    Returns: A reshape tensor of threshold.

    """
    input_shape = input_w.shape
    n_axis = len(input_shape)
    quantization_axis = n_axis + in_quantization_axis if in_quantization_axis < 0 else in_quantization_axis
    reshape_shape = [-1 if i == quantization_axis else 1 for i in range(n_axis)]
    ptq_threshold_tensor = tf.reshape(threshold_tensor, reshape_shape)
    return ptq_threshold_tensor
