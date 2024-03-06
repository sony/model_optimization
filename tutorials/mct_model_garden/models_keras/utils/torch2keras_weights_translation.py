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

from typing import Dict
import tensorflow as tf
import torch
import numpy as np


ln_patch = False


def weight_translation(keras_name: str, pytorch_weights_dict: Dict[str, np.ndarray],
                       layer: tf.keras.layers.Layer) -> np.ndarray:
    """
    Convert a keras weight name format to torch naming format, so the value of the weight can be
    retrieved from the Torch model state_dict.

    For example:
    * Keras name: model_name/layer_name/kernel:0
    is translated to:
    * Torch name: model_name.layer_name.weight

    Args:
        keras_name: keras weight name
        pytorch_weights_dict: the Torch model state_dict, as {name_str: weight value as numpy array}
        layer: the Keras layer of the weight

    Returns:
        the weight value as a numpy array

    """
    keras_name = keras_name.replace('/', '.')
    if ln_patch and (isinstance(layer, tf.keras.layers.LayerNormalization) or
                     (isinstance(layer, tf.keras.layers.BatchNormalization) and '_bn_patch' in layer.name)):
        if isinstance(layer, tf.keras.layers.LayerNormalization):
            if '.beta:0' in keras_name:
                value = layer.weights[1].numpy()
            elif '.gamma:0' in keras_name:
                value = layer.weights[0].numpy()
            else:
                raise Exception('Unknown LayerNorm weight name')
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            if '.beta:0' in keras_name:
                value = pytorch_weights_dict.pop(keras_name.replace('_bn_patch', '').replace(".beta:0", ".bias"))
            elif '.gamma:0' in keras_name:
                value = pytorch_weights_dict.pop(keras_name.replace('_bn_patch', '').replace(".gamma:0", ".weight"))
            elif '.moving_mean:0' in keras_name:
                value = layer.weights[2].numpy()
            elif '.moving_variance:0' in keras_name:
                value = layer.weights[3].numpy()
            else:
                raise Exception('Unknown BatchNorm weight name')
        else:
            raise NotImplemented
    # Handling MHA layers
    elif isinstance(layer, tf.keras.layers.MultiHeadAttention):
        if '.bias:0' in keras_name:
            if '.query.' in keras_name:
                value = pytorch_weights_dict[keras_name.replace(".query.bias:0", ".qkv_proj.bias")]
                value = value[:int(value.shape[0]/3)].reshape((layer._num_heads, -1))
            elif '.key.' in keras_name:
                value = pytorch_weights_dict[keras_name.replace(".key.bias:0", ".qkv_proj.bias")]
                value = value[int(value.shape[0] / 3):2*int(value.shape[0] / 3)].reshape((layer._num_heads, -1))
            elif '.value.' in keras_name:
                value = pytorch_weights_dict[keras_name.replace(".value.bias:0", ".qkv_proj.bias")]
                value = value[2*int(value.shape[0] / 3):].reshape((layer._num_heads, -1))
            elif '.attention_output.' in keras_name:  # or '.key.' in keras_name or '.value.' in keras_name:
                value = pytorch_weights_dict[keras_name.replace(".attention_output.bias:0", ".out_proj.bias")]
            else:
                raise Exception('Unknown MHA bias name')
        elif '.query.' in keras_name:
            value = pytorch_weights_dict[keras_name.replace(".query.kernel:0", ".qkv_proj.weight")]
            value = value[:int(value.shape[0]/3), :].transpose().reshape((int(value.shape[0]/3), layer._num_heads, -1))
        elif '.key.' in keras_name:
            value = pytorch_weights_dict[keras_name.replace(".key.kernel:0", ".qkv_proj.weight")]
            value = value[int(value.shape[0] / 3):2 * int(value.shape[0] / 3), :].transpose().reshape((int(value.shape[0]/3), layer._num_heads, -1))
        elif '.value.' in keras_name:
            value = pytorch_weights_dict[keras_name.replace(".value.kernel:0", ".qkv_proj.weight")]
            value = value[2*int(value.shape[0] / 3):, :].transpose().reshape((int(value.shape[0]/3), layer._num_heads, -1))
        elif '.attention_output.' in keras_name:  # or '.key.' in keras_name or '.value.' in keras_name:
            value = pytorch_weights_dict[keras_name.replace(".attention_output.kernel:0", ".out_proj.weight")]
            value = value.transpose().reshape((layer._num_heads, -1, value.shape[-1]))
        else:
            raise Exception('Unknown MHA weight name')

    # Handle Convolution layers
    elif '.depthwise_kernel:0' in keras_name:
        value = pytorch_weights_dict.pop(keras_name.replace(".depthwise_kernel:0", ".weight")).transpose((2, 3, 0, 1))
    elif '.kernel:0' in keras_name:
        if isinstance(layer, tf.keras.layers.Dense):
            value = pytorch_weights_dict.pop(keras_name.replace(".kernel:0", ".weight")).transpose((1, 0))
        else:
            value = pytorch_weights_dict.pop(keras_name.replace(".kernel:0", ".weight"))
            if len(value.shape) == 2:
                assert layer.kernel_size == (1, 1), "Error: Thie code is for converting Dense kernels to conv1x1"
                value = value.transpose().reshape(layer.kernel._shape_tuple())
            else:
                value = value.transpose((2, 3, 1, 0))
    elif '.bias:0' in keras_name:
        value = pytorch_weights_dict.pop(keras_name.replace(".bias:0", ".bias"))

    # Handle normalization layers
    elif '.beta:0' in keras_name:
        value = pytorch_weights_dict.pop(keras_name.replace(".beta:0", ".bias"))
    elif '.gamma:0' in keras_name:
        value = pytorch_weights_dict.pop(keras_name.replace(".gamma:0", ".weight"))
    elif '.moving_mean:0' in keras_name:
        value = pytorch_weights_dict.pop(keras_name.replace(".moving_mean:0", ".running_mean"))
    elif '.moving_variance:0' in keras_name:
        value = pytorch_weights_dict.pop(keras_name.replace(".moving_variance:0", ".running_var"))
    else:
        value = pytorch_weights_dict.pop(keras_name)
    return value


def load_state_dict(model: tf.keras.Model, state_dict_url: str = None,
                    state_dict_torch: Dict = None):
    """
    Assign a Keras model weights according to a state_dict from the equivalent Torch model.
    Args:
        model: A Keras model
        state_dict_url: the Torch model state_dict location
        state_dict_torch: Torch model state_dict. If not None, will be used instead of state_dict_url

    Returns:
        The same model object after assigning the weights

    """
    if state_dict_torch is None:
        assert state_dict_url is not None, "either 'state_dict_url' or 'state_dict_torch' should not be None"
        state_dict_torch = torch.hub.load_state_dict_from_url(state_dict_url, progress=False,
                                                              map_location='cpu')
    state_dict = {k: v.numpy() for k, v in state_dict_torch.items()}

    for layer in model.layers:
        for w in layer.weights:
            w.assign(weight_translation(w.name, state_dict, layer))

    # look for variables not assigned in torch state dict
    for k in state_dict:
        if 'num_batches_tracked' in k:
            continue
        print(f'  WARNING: {k} not assigned to keras model !!!')
