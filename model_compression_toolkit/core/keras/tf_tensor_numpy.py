# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Union, List, Tuple
import tensorflow as tf
import numpy as np


def to_tf_tensor(tensor):
    """
    Convert a Numpy array to a TF tensor.
    Args:
        tensor: Numpy array.

    Returns:
        TF tensor converted from the input Numpy array.
    """
    if isinstance(tensor, tf.Tensor):
        return tensor
    elif isinstance(tensor, list):
        return [to_tf_tensor(t) for t in tensor]
    elif isinstance(tensor, tuple):
        return (to_tf_tensor(t) for t in tensor)
    elif isinstance(tensor, np.ndarray):
        return tf.convert_to_tensor(tensor.astype(np.float32))
    else:
        raise Exception(f'Conversion of type {type(tensor)} to {type(tf.Tensor)} is not supported')


def tf_tensor_to_numpy(tensor: Union[List, Tuple, np.ndarray, tf.Tensor],
                       is_single_tensor=False) -> np.ndarray:
    """
    Convert a TF tensor to a Numpy array.
    Args:
        tensor: TF tensor.
        is_single_tensor: whether input is a value to be converted to a single tensor.
                          if False, recurse the lists and tuples

    Returns:
        Numpy array converted from the input tensor.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, list):
        if is_single_tensor:
            return np.array(tensor)
        else:
            return [tf_tensor_to_numpy(t) for t in tensor]
    elif isinstance(tensor, tuple):
        if is_single_tensor:
            return np.array(tensor)
        else:
            return (tf_tensor_to_numpy(t) for t in tensor)
    elif isinstance(tensor, tf.Tensor):
        return tensor.numpy()
    else:
        raise Exception(f'Conversion of type {type(tensor)} to {type(np.ndarray)} is not supported')
