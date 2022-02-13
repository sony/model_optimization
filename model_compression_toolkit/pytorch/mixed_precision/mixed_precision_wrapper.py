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

from typing import Any

import torch
import copy

from model_compression_toolkit import FrameworkInfo
from model_compression_toolkit.common import BaseNode
from model_compression_toolkit.pytorch.utils import set_model


class PytorchMixedPrecisionWrapper(torch.nn.Module):
    """
    Class for reconstructing a Pytorch model from a graph
    """
    def __init__(self,
                 n: BaseNode,
                 fw_info: FrameworkInfo):
        """
        Construct a Pytorch model.
        Args:
            graph: Graph to build its corresponding Pytorch model.
            mode: Building mode. Read ModelBuilderMode description for more info.
            append2output: List of nodes or OutTensor objects.
        """
        super(PytorchMixedPrecisionWrapper, self).__init__()

        assert n.candidates_weights_quantization_cfg is not None

        framework_attr = copy.copy(n.framework_attr)
        self.layer = n.type(**framework_attr)
        # loading the weights from the graph node (weights of the trained model)
        self.layer.load_state_dict({k: torch.Tensor(v) for k, v in n.weights.items()}, strict=False)
        set_model(self.layer)

        self.weight_attrs = fw_info.get_kernel_op_attributes(n.type)
        # float_weights is a list of weights for each attribute that we want to quantize.
        self.float_weights = [n.get_weights_by_keys(attr) for attr in
                              self.weight_attrs]

        assert len(self.weight_attrs) == len(self.float_weights)

        self.node_weights_q_cfg = n.candidates_weights_quantization_cfg

        self.quantizer_fn = self.node_weights_q_cfg[0].weights_quantization_fn
        for qc in self.node_weights_q_cfg:
            assert qc.weights_quantization_fn == self.quantizer_fn

        self.quantized_weights = self._get_quantized_weights()

    def forward(self, x) -> Any:
        """
        Args:
            x: input tensors to model.

        Returns:
            torch Tensor which is the output of the model logic.

        """
        return self.layer(x)

    def _get_quantized_weights(self):
        quantized_weights = []
        for qc in self.node_weights_q_cfg:
            # for each quantization configuration in mixed precision
            # get quantized weights for each attribute and for each filter
            quantized_per_attr = []
            for float_weight in self.float_weights:
                # for each attribute
                quantized_per_attr.append(self.quantizer_fn(tensor_data=float_weight,
                                                            n_bits=qc.weights_n_bits,
                                                            signed=True,
                                                            quantization_params=qc.weights_quantization_params,
                                                            per_channel=qc.weights_per_channel_threshold,
                                                            output_channels_axis=qc.weights_channels_axis))
            quantized_weights.append(quantized_per_attr)

        return quantized_weights

    def set_active_weights(self,
                           bitwidth_idx: int,
                           attr: str = None):
        if attr is None:  # set bit width to all weights of the layer
            attr_idxs = [attr_idx for attr_idx in range(len(self.quantized_weights[bitwidth_idx]))]
            self._set_bit_width_index(bitwidth_idx, attr_idxs)
        else:  # set bit width to a specific attribute
            attr_idx = self.weight_attrs.index(attr)
            self._set_bit_width_index(bitwidth_idx, [attr_idx])

    def _set_bit_width_index(self, bitwidth_idx, attr_idxs):
        assert bitwidth_idx < len(self.quantized_weights), \
            f"Index {bitwidth_idx} does not exist in current quantization candidates list"

        loaded_weights = {k: torch.as_tensor(v) for k, v in self.layer.state_dict().items()}
        with torch.no_grad():
            for attr_idx in attr_idxs:
                weights_tensor = self.quantized_weights[bitwidth_idx][attr_idx]
                weights_device = loaded_weights[self.weight_attrs[attr_idx]].device
                active_weights = torch.nn.Parameter(torch.from_numpy(weights_tensor).to(weights_device))
                loaded_weights[self.weight_attrs[attr_idx]] = active_weights
            self.layer.load_state_dict(loaded_weights, strict=True)
