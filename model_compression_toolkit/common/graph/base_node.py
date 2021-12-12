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

import copy
from typing import Dict, Any, Tuple

import numpy as np

from model_compression_toolkit.common.constants import WEIGHTS_NBITS_ATTRIBUTE, CORRECTED_BIAS_ATTRIBUTE


class BaseNode:
    """
    Class to represent a node in a graph that represents the model.
    """

    def __init__(self,
                 name: str,
                 framework_attr: Dict[str, Any],
                 input_shape: Tuple[Any],
                 output_shape: Tuple[Any],
                 weights: Dict[str, np.ndarray],
                 layer_class: type,
                 reuse: bool = False,
                 reuse_group: str = None,
                 quantization_attr: Dict[str, Any] = None):
        """
        Init a Node object.

        Args:
            name: Node's name
            framework_attr: Framework attributes the layer had which the node holds.
            input_shape: Input tensor shape of the node.
            output_shape: Input tensor shape of the node.
            weights: Dictionary from a variable name to the weights with that name in the layer the node represents.
            layer_class: Class path of the layer this node represents.
            reuse: Whether this node was duplicated and represents a reused layer.
            reuse_group: Name of group of nodes from the same reused layer.
            quantization_attr: Attributes the node holds regarding how it should be quantized.
        """
        self.name = name
        self.framework_attr = framework_attr
        self.quantization_attr = quantization_attr if quantization_attr is not None else dict()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = weights
        self.layer_class = layer_class
        self.reuse = reuse
        self.reuse_group = reuse_group
        self.activation_quantization_cfg = None
        self.final_weights_quantization_cfg = None
        self.candidates_weights_quantization_cfg = None
        self.output_quantization = True

    def no_quantization(self) -> bool:
        """

        Returns: Whether NodeQuantizationConfig does not have activation params.

        """
        return self.activation_quantization_cfg is None or \
               (not self.activation_quantization_cfg.has_activation_quantization_params())

    def weight_quantization(self) -> bool:
        """

        Returns: Whether node weights should be quantized

        """
        return self.final_weights_quantization_cfg is not None and \
               self.final_weights_quantization_cfg.has_weights_quantization_params() and \
               self.final_weights_quantization_cfg.enable_weights_quantization

    def activation_quantization(self) -> bool:
        """

        Returns: Whether node activation should be quantized

        """
        return self.activation_quantization_cfg is not None and \
               self.activation_quantization_cfg.has_activation_quantization_params() and \
               self.activation_quantization_cfg.enable_activation_quantization

    def __repr__(self):
        """

        Returns: String that represents the node.

        """
        return f'{self.layer_class.__name__}:{self.name}'

    def get_weights_by_keys(self, name: str) -> np.ndarray:
        """
        Get a node's weight by its name.
        Args:
            name: Name of the variable for a node's weight.

        Returns:
            A node's weight (by its name).
        """
        res = [k for k in self.weights.keys() if name in k]
        if len(res) == 1:  # Make sure there are no duplicates
            return self.weights[res[0]]
        else:
            return None

    def set_weights_by_keys(self, name: str, tensor: np.ndarray):
        """
        Set a new weight to one of the existing node's weights, or add it if not exist.

        Args:
            name: Name of the weight the node holds.
            tensor: Numpy array to set as the weight.

        """

        res = [k for k in self.weights.keys() if name in k]
        if len(res) == 1:
            self.weights[res[0]] = tensor
        else:  # Add if not exist
            self.weights[name] = tensor
            self.weights_keys = list(self.weights.keys())  # update keys

    def get_weights_list(self):
        """

        Returns: A list of all weights the node holds.

        """
        return [self.weights[k] for k in self.weights.keys() if self.weights[k] is not None]

    def get_num_parameters(self) -> int:
        """

        Returns: Number of parameters the node holds.

        """
        node_num_params = np.sum([v.flatten().shape[0] for v in self.weights.values() if v is not None])
        assert int(node_num_params) == node_num_params
        return int(node_num_params)

    def get_memory_bytes(self, by_candidate_idx: int = None) -> float:
        """

        Returns: Number of bytes the node's memory requires.

        """
        params = self.get_num_parameters()
        if by_candidate_idx is not None:
            assert type(by_candidate_idx)==int
            assert by_candidate_idx < len(self.candidates_weights_quantization_cfg)
            memory = params * self.candidates_weights_quantization_cfg[by_candidate_idx].weights_n_bits / 8 # in bytes

        elif self.final_weights_quantization_cfg is None:  # float coefficients
            memory = params * 4

        else:
            memory = params * self.final_weights_quantization_cfg.weights_n_bits / 8  # in bytes

        return memory

    def get_unified_candidates_dict(self):
        """
        In Mixed-Precision, a node can have multiple candidates for weights quantization configuration.
        In order to display a single view of a node (for example, for logging in TensorBoard) we need a way
        to create a single dictionary from all candidates.
        This method is aimed to build such an unified dictionary for a node.

        Returns: A dictionary containing information from node's weight quantization configuration candidates.

        """
        shared_attributes = [CORRECTED_BIAS_ATTRIBUTE, WEIGHTS_NBITS_ATTRIBUTE]
        attr = dict()
        if self.candidates_weights_quantization_cfg is not None:
            attr = copy.deepcopy(self.candidates_weights_quantization_cfg[0].__dict__)
            for shared_attr in shared_attributes:
                if shared_attr in attr:
                    unified_attr = []
                    for candidate in self.candidates_weights_quantization_cfg:
                        unified_attr.append(getattr(candidate, shared_attr))
                    attr[shared_attr] = unified_attr
        return attr

