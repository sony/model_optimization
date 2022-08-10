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

from typing import Any, List

import torch
import copy

from model_compression_toolkit import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.pytorch.utils import set_model, to_torch_tensor


class PytorchMixedPrecisionWrapper(torch.nn.Module):
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
                 fw_info: FrameworkInfo):
        """
        Construct a Pytorch model that constitutes as a wrapper for a Pytorch layer, built from a given graph node.
        Args:
            n: Node to build its Pytorch layer.
            fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).
        """
        super(PytorchMixedPrecisionWrapper, self).__init__()

        assert n.candidates_quantization_cfg is not None
        self.node_q_cfg = n.candidates_quantization_cfg
        if isinstance(n, FunctionalNode):
            self.layer = n.type
        else:
            framework_attr = copy.copy(n.framework_attr)
            self.layer = n.type(**framework_attr)

        for qc in self.node_q_cfg:
            assert qc.weights_quantization_cfg.enable_weights_quantization == \
                   self.node_q_cfg[0].weights_quantization_cfg.enable_weights_quantization \
                   and qc.activation_quantization_cfg.enable_activation_quantization == \
                   self.node_q_cfg[0].activation_quantization_cfg.enable_activation_quantization, \
                "Candidates with different weights/activation enabled properties is currently not supported"

        self.enable_weights_quantization = \
            self.node_q_cfg[0].weights_quantization_cfg.enable_weights_quantization and \
            not n.is_all_weights_candidates_equal()
        self.enable_activation_quantization = \
            self.node_q_cfg[0].activation_quantization_cfg.enable_activation_quantization and \
            not n.is_all_activation_candidates_equal()

        max_cfg_candidates = n.find_max_candidates_indices()
        assert len(max_cfg_candidates) == 1, \
            f"A maximal config candidate must be defined, but some node have multiple potential maximal candidates"
        max_candidate_idx = max_cfg_candidates[0]

        if not isinstance(n, FunctionalNode):
            # loading the weights (if exists) from the graph node (weights of the trained model)
            self.layer.load_state_dict({k: torch.Tensor(v) for k, v in n.weights.items()}, strict=False)
            set_model(self.layer)

        # Setting layers' weights
        if self.enable_weights_quantization:
            self.weight_attrs = fw_info.get_kernel_op_attributes(n.type)
            # float_weights is a list of weights for each attribute that we want to quantize.
            self.float_weights = [n.get_weights_by_keys(attr) for attr in
                                  self.weight_attrs]

            assert len(self.weight_attrs) == len(self.float_weights)
            self.weights_quantizer_fn_list = [qc.weights_quantization_cfg.weights_quantization_fn
                                              for qc in self.node_q_cfg]
            self.quantized_weights = self._get_quantized_weights()
            # Setting the model with the initial quantized weights (the highest precision)
            self.set_active_weights(bitwidth_idx=max_candidate_idx)

        # Setting layer's activation
        if self.enable_activation_quantization:
            self.activation_quantizers = self._get_activation_quantizers()
            self.activation_bitwidth_idx = max_candidate_idx

    def forward(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Args:
            x: input tensors to layer.
        Returns:
            torch Tensor which is the output of the wrapped layer on the given input.
        """
        outputs = self.layer(x, *args, **kwargs)

        if self.enable_activation_quantization:
            # add fake quant to quantize activations with the active number of bits
            if isinstance(outputs, list):
                # we assume here that it can't be multiple outputs out of a quantized layer
                assert len(outputs) == 1, "Activation quantization for node with multiple outputs is not supported."
                outputs = torch.cat(outputs, dim=0)

            outputs = self.activation_quantizers[self.activation_bitwidth_idx](outputs)

        return outputs

    def _get_quantized_weights(self):
        """
        Calculates the quantized weights' tensors for each of the bitwidth candidates for quantization,
        to be stored and used during MP search.
        Returns: a list of quantized weights - for each bitwidth and layer's attribute to be quantized.
        """
        quantized_weights = []
        for index, qc in enumerate(self.node_q_cfg):
            # for each quantization configuration in mixed precision
            # get quantized weights for each attribute and for each filter
            quantized_per_attr = []
            for float_weight in self.float_weights:
                # for each attribute
                quantized_per_attr.append(self.weights_quantizer_fn_list[index](tensor_data=float_weight,
                                                                                n_bits=qc.weights_quantization_cfg.weights_n_bits,
                                                                                signed=True,
                                                                                quantization_params=qc.weights_quantization_cfg.weights_quantization_params,
                                                                                per_channel=qc.weights_quantization_cfg.weights_per_channel_threshold,
                                                                                output_channels_axis=qc.weights_quantization_cfg.weights_channels_axis))
            quantized_weights.append(quantized_per_attr)

        return quantized_weights

    def _get_activation_quantizers(self) -> List[Any]:
        """
        Builds a list of quantizers for each of the bitwidth candidates for activation quantization,
        to be stored and used during MP search.

        Returns: a list of activation quantizers - for each bitwidth and layer's attribute to be quantized.
        """
        activation_quantizers = []
        for index, qc in enumerate(self.node_q_cfg):
            q_activation = self.node_q_cfg[index].activation_quantization_cfg
            activation_quantizers.append(q_activation.quantize_node_output)

        return activation_quantizers

    def set_active_weights(self,
                           bitwidth_idx: int,
                           attr: str = None):
        """
        Set a weights' tensor to use by the layer wrapped by the module.
        Args:
            bitwidth_idx: Index of a candidate quantization configuration to use its quantized
            version of the float weight.
            attr: Attributes of the layer's weights to quantize
        """
        if self.enable_weights_quantization:
            if attr is None:  # set bit width to all weights of the layer
                attr_idxs = [attr_idx for attr_idx in range(len(self.quantized_weights[bitwidth_idx]))]
                self._set_weights_bit_width_index(bitwidth_idx, attr_idxs)
            else:  # set bit width to a specific attribute
                attr_idx = self.weight_attrs.index(attr)
                self._set_weights_bit_width_index(bitwidth_idx, [attr_idx])

    def set_active_activation_quantizer(self,
                                        bitwidth_idx: int):
        """
        Set an activation quantizer to use by the layer wrapped by the module.

        Args:
            bitwidth_idx: Index of a candidate quantization configuration to use its quantizer
            for quantizing the activation.
        """
        if self.enable_activation_quantization:
            self.activation_bitwidth_idx = bitwidth_idx

    def _set_weights_bit_width_index(self,
                                     bitwidth_idx: int,
                                     attr_idxs: List[int]):
        """
        Sets the wrapped layer's weights state with quantized weights, according to the given configuration.
        Args:
            bitwidth_idx: Index of a candidate quantization configuration to use its quantized
            version of the float weight.
            attr_idxs: Indices list of attributes of the layer's weights to quantize
        Returns: None (sets the new state of the layer inplace).
        """
        assert bitwidth_idx < len(self.quantized_weights), \
            f"Index {bitwidth_idx} does not exist in current quantization candidates list"

        loaded_weights = {k: torch.as_tensor(v) for k, v in self.layer.state_dict().items()}
        with torch.no_grad():
            for attr_idx in attr_idxs:
                # need to prepare the weights' tensor - extract it from the maintained quantized_weights list
                # and move it to the relevant device as the wrapped layer's weights.
                weights_tensor = self.quantized_weights[bitwidth_idx][attr_idx]
                weights_device = loaded_weights[self.weight_attrs[attr_idx]].device
                active_weights = torch.nn.Parameter(torch.from_numpy(weights_tensor).to(weights_device))
                loaded_weights[self.weight_attrs[attr_idx]] = active_weights
            self.layer.load_state_dict(loaded_weights, strict=True)
