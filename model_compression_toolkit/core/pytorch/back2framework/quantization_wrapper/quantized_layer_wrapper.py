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

import copy
import torch
from typing import Any

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.pytorch.back2framework.quantization_wrapper.wrapper_quantize_config import \
    WrapperQuantizeConfig
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.utils import set_model




class QuantizedLayerWrapper(torch.nn.Module):
    """
    Class that wraps a Pytorch layer (nn.Module) to be used for mixed precision quantization.
    Allows to maintain quantized weights tensors for each of the layer's attributes that we want to quantize,
    and a list of activation quantizers for each quantization candidate,
    for each of the candidate bitwidth options specified for the mixed precision model.
    During MP search, it allows to activate the relevant quantized weights tensor and activation quantizer
    according to a given configuration, and use it for inference.
    """
    def __init__(self,
                 n: BaseNode,
                 quantize_config:WrapperQuantizeConfig):
        """
        Construct a Pytorch model that constitutes as a wrapper for a Pytorch layer, built from a given graph node.
        Args:
            n: Node to build its Pytorch layer.

        """
        super(QuantizedLayerWrapper, self).__init__()

        self.is_function = isinstance(n, FunctionalNode)

        # Build a layer and load its weights
        self._build_layer(n)
        self.quantization_config = quantize_config

        # Setting layers' weights
        if self.quantization_config.is_weight_quantized and not self.is_function: #TODO: can functions have weights
            self._quantize_weights(n)

        if not self.is_function:
            set_model(self.layer)


    def forward(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Args:
            x: input tensors to layer.
        Returns:
            torch Tensor which is the output of the wrapped layer on the given input.
        """
        outputs = self.layer(x, *args, **kwargs)

        if self.quantization_config.is_activation_quantized:
            if isinstance(outputs, list):
                outputs = torch.cat(outputs, dim=0)
            outputs = self.quantization_config.get_activation_quantizers()[0](outputs)

        return outputs



    def _build_layer(self, n):
        if self.is_function:
            self.layer = n.type
        else:
            framework_attr = copy.copy(n.framework_attr)
            self.layer = n.type(**framework_attr)
            self.layer.load_state_dict({k: torch.Tensor(v) for k, v in n.weights.items()}, strict=False)

    def _quantize_weights(self, n):
        self.weight_attrs = DEFAULT_PYTORCH_INFO.get_kernel_op_attributes(n.type)
        # float_weights is a list of weights for each attribute that we want to quantize.
        float_weights = [n.get_weights_by_keys(attr) for attr in self.weight_attrs]
        assert len(self.weight_attrs) == len(float_weights)
        self.quantized_weights = self._get_quantized_weights(float_weights)
        if not isinstance(n, FunctionalNode):
            self._load_quantized_weights()

    def _load_quantized_weights(self):
        # loading the weights (if exists) from the graph node (weights of the trained model)
        assert len(self.weight_attrs) == len(self.quantized_weights)
        loaded_weights = {k: torch.as_tensor(v) for k, v in self.layer.state_dict().items()}
        with torch.no_grad():
            for attr_idx, attr in enumerate(self.weight_attrs):
                # need to prepare the weights' tensor - extract it from the maintained quantized_weights list
                # and move it to the relevant device as the wrapped layer's weights.
                weights_tensor = self.quantized_weights[attr_idx]
                weights_device = loaded_weights[self.weight_attrs[attr_idx]].device
                active_weights = torch.nn.Parameter(torch.from_numpy(weights_tensor).to(weights_device))
                loaded_weights[self.weight_attrs[attr_idx]] = active_weights
            self.layer.load_state_dict(loaded_weights, strict=True)


    def _get_quantized_weights(self, float_weights):
        """
        Calculates the quantized weights' tensors for each of the bitwidth candidates for quantization,
        to be stored and used during MP search.
        Returns: a list of quantized weights - for each bitwidth and layer's attribute to be quantized.
        """

        quantized_weights = []
        quantizers = self.quantization_config.get_weight_quantizers()
        assert len(quantizers)==len(float_weights)
        for float_weight, quantizer in zip(float_weights, quantizers):
            # for each attribute
            quantized_weights.append(quantizer(float_weight=float_weight))
        return quantized_weights


