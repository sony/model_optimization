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

import numpy as np
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Conv2DTranspose, Dense

from sony_model_optimization_package.common import Node
from sony_model_optimization_package.keras.constants import KERNEL


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
