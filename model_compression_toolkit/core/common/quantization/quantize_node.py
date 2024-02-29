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


from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.common.quantization.node_quantization_config import WeightsAttrQuantizationConfig


def get_quantized_weights_attr_by_qc(attr_name: str,
                                     n: BaseNode,
                                     weights_qc: WeightsAttrQuantizationConfig):
    """
    For a weights attribute and weights attribute quantization configuration, compute
    the quantized weights of the node's attribute and return it
    and the input/output channels indices (if relevant, o.w. None).

    Args:
        attr_name: The name of the attribute to quantize.
        n: Node to quantize its weights attribute.
        weights_qc: Weight attribute quantization configuration to use for the quantization.

    Returns:
        A quantized kernel of the node using a weights quantization configuration.
    """

    channels_axis = weights_qc.weights_channels_axis
    if channels_axis is not None:
        # switching output and input channel axis order in the tuple because this is what
        # the caller of this function expect. The new order is: (input, output)
        channels_axis = (channels_axis[1], channels_axis[0])
        output_channels_axis = channels_axis[1]
    else:
        channels_axis = None
        output_channels_axis = None

    Logger.debug(f'quantizing layer {n.name} attribute {attr_name} with {weights_qc.weights_n_bits} bits')
    quantized_kernel = weights_qc.weights_quantization_fn(n.get_weights_by_keys(attr_name),
                                                          n_bits=weights_qc.weights_n_bits,
                                                          signed=True,
                                                          quantization_params=weights_qc.weights_quantization_params,
                                                          per_channel=weights_qc.weights_per_channel_threshold,
                                                          output_channels_axis=output_channels_axis)

    return quantized_kernel, channels_axis
