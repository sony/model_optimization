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
from typing import Dict, Any

import numpy as np

from model_compression_toolkit.constants import NUM_QPARAM_HESSIAN_SAMPLES
from model_compression_toolkit.core.common.hessian import HessianInfoService
from model_compression_toolkit.defaultdict import DefaultDict
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig, \
    WeightsAttrQuantizationConfig

# If the quantization config does not contain kernel channel mapping or the weights
# quantization is not per-channel, we use a dummy channel mapping.
dummy_channel_mapping = DefaultDict(default_value=(None, None))


def get_weights_qparams(weights_attr_values: np.ndarray,
                        weights_quant_config: NodeWeightsQuantizationConfig,
                        attr_quant_config: WeightsAttrQuantizationConfig,
                        output_channels_axis: int,
                        node=None,
                        hessian_info_service: HessianInfoService = None,
                        num_hessian_samples: int = NUM_QPARAM_HESSIAN_SAMPLES) -> Dict[Any, Any]:
    """
    Compute thresholds to quantize a kernel according to a NodeWeightsQuantizationConfig
    instance.

    Args:
        weights_attr_values: Weights attribute parameter to compute the quantization thresholds for.
        weights_quant_config: Weights quantization configuration to define how the thresholds are computed.
        attr_quant_config: A specific weights attribute quantization configuration to get its params.
        output_channels_axis: Index of the kernel output channels dimension.
        node: The node for which the quantization error is computed (used only with HMSE error method).
        hessian_info_service: HessianInfoService object for retrieving Hessian-based scores (used only with HMSE error method).
        num_hessian_samples: Number of samples to approximate Hessian-based scores on (used only with HMSE error method).

    Returns:
        A dictionary with the quantization threshold of the kernel.
    """
    if attr_quant_config.weights_quantization_params_fn is not None:
        weights_params = attr_quant_config.weights_quantization_params_fn(weights_attr_values,
                                                                          p=attr_quant_config.l_p_value,
                                                                          n_bits=attr_quant_config.weights_n_bits,
                                                                          per_channel=attr_quant_config.weights_per_channel_threshold and output_channels_axis is not None,
                                                                          channel_axis=output_channels_axis,
                                                                          min_threshold=weights_quant_config.min_threshold,
                                                                          quant_error_method=attr_quant_config.weights_error_method,
                                                                          node=node,
                                                                          hessian_info_service=hessian_info_service,
                                                                          num_hessian_samples=num_hessian_samples)
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
