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

import numpy as np
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Conv2DTranspose, Dense

from model_compression_toolkit.common import Node
from model_compression_toolkit.keras.constants import KERNEL


def node_to_bops(n: Node):
    """
    Compute the number of BOPs for a node in a graph from a Keras model.
    For now, it computes the number of BOPs for Conv2D, Conv2DTranspose, DepthwiseConv2D and Dense.
    For other layers, it returns 0.

    Args:
        n: Node to compute its number of BOPs.

    Returns:
        Number of BOPs of a node in a graph for a Keras model.
    """

    macs = 0
    if n.layer_class in [Conv2D, DepthwiseConv2D, Conv2DTranspose]:
        kernel_shape = n.get_weights_by_keys(KERNEL).shape
        if n.framework_attr.get('data_format') == 'channels_last':
            out_w, out_h = n.output_shape[1], n.output_shape[2]
        else:
            out_w, out_h = n.output_shape[2], n.output_shape[3]

        macs = out_w * out_h * np.prod(kernel_shape)

    elif n.layer_class in [Dense]:
        macs = np.prod(n.get_weights_by_keys(KERNEL).shape)

    if n.weights_quantization_cfg is None or n.activation_quantization_cfg is None:
        bops = macs * 32 * 32
    else:
        bops = macs * n.weights_quantization_cfg.weights_n_bits * n.activation_quantization_cfg.activation_n_bits

    return bops
