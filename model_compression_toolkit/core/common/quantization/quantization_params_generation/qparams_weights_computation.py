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
from typing import Dict, Any, Tuple

import numpy as np

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.defaultdict import DefaultDict
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig


# If the quantization config does not contain kernel channel mapping or the weights
# quantization is not per-channel, we use a dummy channel mapping.
dummy_channel_mapping = DefaultDict({}, lambda: (None, None))


def get_weights_qparams(kernel: np.ndarray,
                        weights_quant_config: NodeWeightsQuantizationConfig,
                        output_channels_axis: int) -> Dict[Any, Any]:
    """
    Compute thresholds to quantize a kernel according to a NodeWeightsQuantizationConfig
    instance.

    Args:
        kernel: Kernel to compute the quantization thresholds to.
        weights_quant_config: Weights quantization configuration to define how the thresholds are computed.
        output_channels_axis: Index of the kernel output channels dimension.

    Returns:
        A dictionary with the quantization threshold of the kernel.
    """
    if weights_quant_config.weights_quantization_params_fn is not None:
        weights_params = weights_quant_config.weights_quantization_params_fn(kernel,
                                                                             p=weights_quant_config.l_p_value,
                                                                             n_bits=weights_quant_config.weights_n_bits,
                                                                             per_channel=weights_quant_config.weights_per_channel_threshold and output_channels_axis is not None,
                                                                             channel_axis=output_channels_axis,
                                                                             min_threshold=weights_quant_config.min_threshold,
                                                                             quant_error_method=weights_quant_config.weights_error_method)
    else:
        weights_params = {}

    return weights_params




def _get_kernel_channels_mapping(fw_info:FrameworkInfo,
                                use_dummy: bool) -> DefaultDict:
    """
    Get a kernel channel mapping from the framework info, or use dummy mapping (which returns a
    tuple of Nones) if use_use_dummy is True.

    Args:
        fw_info: Framework info which contains a kernel channels mapping.
        use_dummy: Whether to use a dummy mapping or not.

    Returns:
        Kernel channels mapping.
    """

    # Set a kernel channels mapping
    if use_dummy:  # If kernel mapping is missing, we use a dummy channels mapping
        kernel_channels_mapping = dummy_channel_mapping
    else:
        kernel_channels_mapping = fw_info.kernel_channels_mapping
    return kernel_channels_mapping




def get_channels_axis(weights_quant_config: NodeWeightsQuantizationConfig,
                      fw_info: FrameworkInfo,
                      node_type: type) -> Tuple[Any, Any]:
    """
    Get the layer's kernel channels input/output indices.

    Args:
        weights_quant_config: NodeWeightsQuantizationConfig object of the node we would like get
        channels axis for. This is needed for whether to use dummy mapping or not.
        fw_info: Framework info contains the kernel channels mapping.
        node_type: Class to get its kernel's channels indices.

    Returns:
        Class's kernel input/output channels indices.
    """
    # If weights should be quantized per-channel but a kernel channels mapping is missing.
    if weights_quant_config.weights_per_channel_threshold and \
            fw_info.kernel_channels_mapping is None:
        Logger.warning('Weights Per Channel Quantization requires channel mapping function,'
                       ' but framework info does not contain one')
    use_dummy = not weights_quant_config.weights_per_channel_threshold and not \
        weights_quant_config.weights_bias_correction
    kernel_channels_mapping = _get_kernel_channels_mapping(fw_info, use_dummy)
    output_channels_axis, input_channels_axis = kernel_channels_mapping.get(node_type)
    return output_channels_axis, input_channels_axis

