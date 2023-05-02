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

import copy
from typing import Dict, Any, Tuple, List

import numpy as np

from model_compression_toolkit.constants import WEIGHTS_NBITS_ATTRIBUTE, CORRECTED_BIAS_ATTRIBUTE, \
    ACTIVATION_NBITS_ATTRIBUTE
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationConfigOptions, \
    TargetPlatformCapabilities, LayerFilterParams


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
        if self.final_activation_quantization_cfg:
            # if we have a final configuration, then we only care to check if it enables activation quantization
            return self.final_activation_quantization_cfg.enable_activation_quantization

        for qc in self.candidates_quantization_cfg:
            assert self.candidates_quantization_cfg[0].activation_quantization_cfg.enable_activation_quantization == \
                   qc.activation_quantization_cfg.enable_activation_quantization
        return self.candidates_quantization_cfg[0].activation_quantization_cfg.enable_activation_quantization

    def is_weights_quantization_enabled(self) -> bool:
        """

        Returns: Whether node weights quantization is enabled or not.

        """
        if self.final_weights_quantization_cfg:
            # if we have a final configuration, then we only care to check if it enables weights quantization
            return self.final_weights_quantization_cfg.enable_weights_quantization

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
        if name is None:
            return None

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

    def get_float_memory_bytes(self, fw_info) -> float:
        """
        Compute the number of bytes the node's memory requires.

        Args:
            fw_info: Framework info to decide which attributes should be quantized.

        Returns: Number of bytes the node's memory requires when in floating point (32 bit).

        """
        q_params, f_params = self.get_num_parameters(fw_info)
        return (f_params + q_params) * 32 / 8 # in bytes

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

    def get_total_output_params(self) -> float:
        """
        Calculates the output size of the node.

        Returns: Output size.
        """
        output_shapes = self.output_shape if isinstance(self.output_shape, List) else [self.output_shape]

        # remove batch size (first element) from output shape
        output_shapes = [s[1:] for s in output_shapes]

        return sum([np.prod([x for x in output_shape if x is not None]) for output_shape in output_shapes])

    def get_total_input_params(self) -> float:
        """
        Calculates the total parameters in the node's input tensors.

        Returns: Input size (i.e., total number of parameters).
        """

        input_shapes = self.input_shape if isinstance(self.input_shape, List) else [self.input_shape]

        # remove batch size (first element) from input shape
        input_shapes = [s[1:] for s in input_shapes]

        return sum([np.prod([x for x in input_shape if x is not None]) for input_shape in input_shapes])

    def find_min_candidates_indices(self) -> List[int]:
        """
        Returns a list with potential minimal candidates.
        A potential minimal candidate is a candidate which its weights_n_bits and activation_n_bits pair is
        on the Pareto Front, i.e., there is no other candidates that its n_bits pair exceeds in both entries.

        Returns: A list of indices of potential minimal candidates.

        """

        # We assume that the candidates are sorted according to weights_n_bits first and activation_n_bits second
        # First, we add the last candidate to the set of minimal candidates (candidate, index)
        first_min = (len(self.candidates_quantization_cfg) - 1,
                     self.candidates_quantization_cfg[-1].activation_quantization_cfg.activation_n_bits)
        min_candidates = [first_min]

        # Iterate over all other candidates, and add ones with higher weights_n_bits but smaller activation_n_bits
        for i, c in reversed(list(enumerate(self.candidates_quantization_cfg))):
            if c.activation_quantization_cfg.activation_n_bits < first_min[1]:
                min_candidates.append((i, c))

        return [i for i, a_n_bits in min_candidates]

    def find_max_candidates_indices(self) -> List[int]:
        """
        Returns a list with potential maximal candidates.
        A potential maximal candidate is a candidate which its weights_n_bits and activation_n_bits pair is
        on the Pareto Front, i.e., there is no other candidates that its n_bits pair is lower in both entries.

        Returns: A list of indices of potential maximal candidates.
        """

        # We assume that the candidates are sorted according to weights_n_bits first and activation_n_bits second
        # First, we add the first candidate to the set of maximal candidates (candidate, index)
        first_max = (0, self.candidates_quantization_cfg[0].activation_quantization_cfg.activation_n_bits)
        max_candidates = [first_max]

        # Iterate over all other candidates, and add ones with higher weights_n_bits but smaller activation_n_bits
        for i, c in enumerate(self.candidates_quantization_cfg):
            if c.activation_quantization_cfg.activation_n_bits > first_max[1]:
                max_candidates.append((i, c))

        return [i for i, a_n_bits in max_candidates]

    def get_unique_weights_candidates(self) -> List[Any]:
        """
        Returns a list with node's candidates of unique weights bit-width value.
        If the node have multiple candidates with the same weights bit-width,
        the first candidate in the list is returned.

        Returns: A list with node's candidates of unique weights bit-width value.
        """

        unique_candidates = copy.deepcopy(self.candidates_quantization_cfg)
        seen_candidates = set()
        unique_candidates = [candidate for candidate in unique_candidates if
                             candidate.weights_quantization_cfg not in seen_candidates
                             and not seen_candidates.add(candidate.weights_quantization_cfg)]
        return unique_candidates

    def get_unique_activation_candidates(self) -> List[Any]:
        """
        Returns a list with node's candidates of unique activation bit-width value.
        If the node have multiple candidates with the same activation bit-width,
        the first candidate in the list is returned.

        Returns: A list with node's candidates of unique activation bit-width value.
        """

        unique_candidates = copy.deepcopy(self.candidates_quantization_cfg)
        seen_candidates = set()
        unique_candidates = [candidate for candidate in unique_candidates if
                             candidate.activation_quantization_cfg not in seen_candidates
                             and not seen_candidates.add(candidate.activation_quantization_cfg)]
        return unique_candidates

    def has_weights_quantization_enabled_candidate(self) -> bool:
        """
        Checks whether the node has quantization configuration candidates that enable weights quantization.

        Returns: True if the node has at list one quantization configuration candidate with weights quantization enabled.
        """

        return len(self.candidates_quantization_cfg) > 0 and \
               any([c.weights_quantization_cfg.enable_weights_quantization for c in self.candidates_quantization_cfg])

    def has_activation_quantization_enabled_candidate(self) -> bool:
        """
        Checks whether the node has quantization configuration candidates that enable activation quantization.

        Returns: True if the node has at list one quantization configuration candidate with activation quantization enabled.
        """

        return len(self.candidates_quantization_cfg) > 0 and \
               any([c.activation_quantization_cfg.enable_activation_quantization for c in self.candidates_quantization_cfg])

    def get_qco(self, tpc: TargetPlatformCapabilities) -> QuantizationConfigOptions:
        """
        Get the QuantizationConfigOptions of the node according
        to the mappings from layers/LayerFilterParams to the OperatorsSet in the TargetPlatformModel.

        Args:
            tpc: TPC to extract the QuantizationConfigOptions for the node

        Returns:
            QuantizationConfigOptions of the node.
        """

        if tpc is None:
            Logger.error(f'Can not retrieve QC options for None TPC')  # pragma: no cover

        for fl, qco in tpc.filterlayer2qco.items():
            if self.is_match_filter_params(fl):
                return qco
        if self.type in tpc.layer2qco:
            return tpc.layer2qco.get(self.type)
        return tpc.tp_model.default_qco


    def is_match_filter_params(self, layer_filter_params: LayerFilterParams) -> bool:
        """
        Check if the node matches a LayerFilterParams according to its
        layer, conditions and keyword-arguments.

        Args:
            layer_filter_params: LayerFilterParams to check if the node matches its properties.

        Returns:
            Whether the node matches to the LayerFilterParams properties.
        """
        # Check the node has the same type as the layer in LayerFilterParams
        if layer_filter_params.layer != self.type:
            return False

        # Get attributes from node to filter
        layer_config = self.framework_attr
        if hasattr(self, "op_call_kwargs"):
            layer_config.update(self.op_call_kwargs)

        for attr, value in layer_filter_params.kwargs.items():
            if layer_config.get(attr) != value:
                return False

        for c in layer_filter_params.conditions:
            if not c.match(layer_config):
                return False

        return True