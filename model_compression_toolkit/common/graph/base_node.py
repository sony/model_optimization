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

from model_compression_toolkit.common.constants import WEIGHTS_NBITS_ATTRIBUTE, CORRECTED_BIAS_ATTRIBUTE, \
    ACTIVATION_NBITS_ATTRIBUTE


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
                 quantization_attr: Dict[str, Any] = None,
                 has_activation: bool = True
                 ):
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
            has_activation: Whether the node has activations that we might want to quantize.
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
        self.final_weights_quantization_cfg = None
        self.final_activation_quantization_cfg = None
        self.candidates_quantization_cfg = None
        self.prior_info = None
        self.has_activation = has_activation

    @property
    def type(self):
        """
        A function to get the node's layer_class op for convenient comparison
        :return: the node's layer_class
        """
        return self.layer_class

    def get_has_activation(self):
        """
        Returns has_activation attribute.

        Returns: Whether the node has activation to quantize.

        """
        return self.has_activation

    def is_activation_quantization_enabled(self) -> bool:
        """

        Returns: Whether node activation quantization is enabled or not.

        """
        for qc in self.candidates_quantization_cfg:
            assert self.candidates_quantization_cfg[0].activation_quantization_cfg.enable_activation_quantization == \
                   qc.activation_quantization_cfg.enable_activation_quantization
        return self.candidates_quantization_cfg[0].activation_quantization_cfg.enable_activation_quantization

    def is_weights_quantization_enabled(self) -> bool:
        """

        Returns: Whether node weights quantization is enabled or not.

        """
        for qc in self.candidates_quantization_cfg:
            assert self.candidates_quantization_cfg[0].weights_quantization_cfg.enable_weights_quantization == \
                   qc.weights_quantization_cfg.enable_weights_quantization
        return self.candidates_quantization_cfg[0].weights_quantization_cfg.enable_weights_quantization

    def __repr__(self):
        """

        Returns: String that represents the node.

        """
        return f'{self.type.__name__}:{self.name}'

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

    def get_num_parameters(self, fw_info) -> Tuple[int,int]:
        """
        Compute the number of parameters the node holds.
        It returns a tuple: Number of quantized parameters, number of float parameters.

        Args:
            fw_info: Framework info to decide which attributes should be quantized.

        Returns:
            A tuple of (Number of quantized parameters, number of float parameters).

        """
        total_node_params = np.sum([w.flatten().shape[0] for w in self.weights.values() if w is not None])

        q_node_num_params = 0

        for attr in fw_info.get_kernel_op_attributes(self.type):
            if attr is not None:
                w = self.get_weights_by_keys(attr)
                if w is not None:
                    q_node_num_params += w.flatten().shape[0]

        f_node_num_params = total_node_params - q_node_num_params

        assert int(q_node_num_params) == q_node_num_params
        assert int(f_node_num_params) == f_node_num_params
        return int(q_node_num_params), int(f_node_num_params)

    def get_memory_bytes(self, fw_info) -> float:
        """
        Compute the number of bytes the node's memory requires.

        Args:
            fw_info: Framework info to decide which attributes should be quantized.

        Returns: Number of bytes the node's memory requires.

        """
        q_params, f_params = self.get_num_parameters(fw_info)
        if self.final_weights_quantization_cfg is None:  # float coefficients
            memory = (f_params+q_params) * 4
        else:
            memory = (f_params*4)+ (q_params * self.final_weights_quantization_cfg.weights_n_bits / 8)  # in bytes

        return memory

    def get_unified_weights_candidates_dict(self):
        """
        In Mixed-Precision, a node can have multiple candidates for weights quantization configuration.
        In order to display a single view of a node (for example, for logging in TensorBoard) we need a way
        to create a single dictionary from all candidates.
        This method is aimed to build such an unified dictionary for a node.

        Returns: A dictionary containing information from node's weight quantization configuration candidates.

        """
        shared_attributes = [CORRECTED_BIAS_ATTRIBUTE, WEIGHTS_NBITS_ATTRIBUTE]
        attr = dict()
        if self.is_weights_quantization_enabled():
            attr = copy.deepcopy(self.candidates_quantization_cfg[0].weights_quantization_cfg.__dict__)
            for shared_attr in shared_attributes:
                if shared_attr in attr:
                    unified_attr = []
                    for candidate in self.candidates_quantization_cfg:
                        unified_attr.append(getattr(candidate.weights_quantization_cfg, shared_attr))
                    attr[shared_attr] = unified_attr
        return attr

    def get_unified_activation_candidates_dict(self):
        """
        In Mixed-Precision, a node can have multiple candidates for activation quantization configuration.
        In order to display a single view of a node (for example, for logging in TensorBoard) we need a way
        to create a single dictionary from all candidates.
        This method is aimed to build such an unified dictionary for a node.

        Returns: A dictionary containing information from node's activation quantization configuration candidates.

        """
        shared_attributes = [ACTIVATION_NBITS_ATTRIBUTE]
        attr = dict()
        if self.is_weights_quantization_enabled():
            attr = copy.deepcopy(self.candidates_quantization_cfg[0].activation_quantization_cfg.__dict__)
            for shared_attr in shared_attributes:
                if shared_attr in attr:
                    unified_attr = []
                    for candidate in self.candidates_quantization_cfg:
                        unified_attr.append(getattr(candidate.activation_quantization_cfg, shared_attr))
                    attr[shared_attr] = unified_attr
        return attr

    def is_all_activation_candidates_equal(self):
        """
        Checks whether all candidates' quantization configuration have the same activation configuration,
        using the self-implemented __eq__ method of class NodeActivationQuantizationConfig.

        Returns: True if all candidates have same activation configuration, False otherwise.

        """
        return all(candidate.activation_quantization_cfg ==
                   self.candidates_quantization_cfg[0].activation_quantization_cfg
                   for candidate in self.candidates_quantization_cfg)

    def is_all_weights_candidates_equal(self):
        """
        Checks whether all candidates' quantization configuration have the same weights configuration,
        using the self-implemented __eq__ method of class NodeWeightsQuantizationConfig.

        Returns: True if all candidates have same weights configuration, False otherwise.

        """
        return all(candidate.weights_quantization_cfg ==
                   self.candidates_quantization_cfg[0].weights_quantization_cfg
                   for candidate in self.candidates_quantization_cfg)

    def has_weights_to_quantize(self, fw_info):
        """
        Checks whether the node has weights that need to be quantized according to the framework info.
        Args:
            fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize).
        Returns: Whether the node has weights that need to be quantized.
        """
        attrs = fw_info.get_kernel_op_attributes(self.type)
        for attr in attrs:
            if attr and self.get_weights_by_keys(attr) is not None:
                return True
        return False
