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
from typing import Dict, Any, Tuple, List, Type, Union

import numpy as np

from model_compression_toolkit.constants import WEIGHTS_NBITS_ATTRIBUTE, CORRECTED_BIAS_ATTRIBUTE, \
    ACTIVATION_N_BITS_ATTRIBUTE, FP32_BYTES_PER_PARAMETER
from model_compression_toolkit.core.common.quantization.node_quantization_config import WeightsAttrQuantizationConfig, \
    ActivationQuantizationMode
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import QuantizationConfigOptions, \
    OpQuantizationConfig
from model_compression_toolkit.target_platform_capabilities.schema.schema_functions import max_input_activation_n_bits
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework import LayerFilterParams
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.framework_quantization_capabilities import \
    FrameworkQuantizationCapabilities


WeightAttrT = Union[str, int]


class BaseNode:
    """
    Class to represent a node in a graph that represents the model.
    """

    def __init__(self,
                 name: str,
                 framework_attr: Dict[str, Any],
                 input_shape: Tuple[Any],
                 output_shape: Tuple[Any],
                 weights: Dict[WeightAttrT, np.ndarray],
                 layer_class: type,
                 reuse: bool = False,
                 reuse_group: str = None,
                 inputs_as_list: bool = False,
                 quantization_attr: Dict[str, Any] = None,
                 has_activation: bool = True,
                 is_custom: bool = False
                 ):
        """
        Init a Node object.

        Args:
            name: Node's name
            framework_attr: Framework attributes the layer had which the node holds.
            input_shape: Input tensor shape of the node.
            output_shape: Input tensor shape of the node.
            weights: Dictionary from a variable name to the weights with that name in the layer the node represents.
                     Constant inputs to a node are also saved in the weights (AKA positional weights) dictionary and
                     their key is their position (an integer) in the node's call_args.
            layer_class: Class path of the layer this node represents.
            reuse: Whether this node was duplicated and represents a reused layer.
            reuse_group: Name of group of nodes from the same reused layer.
            inputs_as_list: Whether to pass the node its input tensors as a list or not when calling the layer.
            quantization_attr: Attributes the node holds regarding how it should be quantized.
            has_activation: Whether the node has activations that we might want to quantize.
            is_custom: Whether the node is custom layer or not.
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
        self.inputs_as_list = inputs_as_list
        self.final_weights_quantization_cfg = None
        self.final_activation_quantization_cfg = None
        self.candidates_quantization_cfg = None
        self.prior_info = None
        self.has_activation = has_activation
        self.is_custom = is_custom

    @property
    def type(self):
        """
        A function to get the node's layer_class op for convenient comparison
        Returns:
            the node's layer_class
        """
        return self.layer_class

    def get_has_activation(self):
        """
        Returns has_activation attribute.

        Returns: Whether the node has activation to quantize.

        """
        return self.has_activation

    @property
    def has_positional_weights(self):
        """
        Returns has_positional_weights attribute.

        Returns: Whether the node has positional weights.

        """
        return any(isinstance(key, int) for key in self.weights.keys())

    def _is_single_quant_mode(self, q_mode: ActivationQuantizationMode) -> bool:
        """ Check whether all candidates have the same unique quantization mode, and if it is 'q_mode'. """

        if self.final_activation_quantization_cfg:
            # if we have a final configuration, then we only care to check if it enables activation quantization.
            return self.final_activation_quantization_cfg.quant_mode == q_mode

        q_modes = {qc.activation_quantization_cfg.quant_mode for qc in self.candidates_quantization_cfg}
        assert len(q_modes) == 1
        return q_modes.pop() == q_mode

    def is_activation_quantization_enabled(self) -> bool:
        """
        Returns: Whether node activation quantization is enabled or not.
        """
        return self._is_single_quant_mode(ActivationQuantizationMode.QUANT)

    def is_quantization_preserving(self) -> bool:
        """
        Returns: Whether node activation quantization information is preserved from its inputs.
        """
        return self._is_single_quant_mode(ActivationQuantizationMode.PRESERVE_QUANT)

    def is_weights_quantization_enabled(self, attr_name: str) -> bool:
        """
        Checks whether a node's weights attribute quantization is enabled.

        Args:
            attr_name: An attribute to check if its quantization is enabled.

        Returns: Whether node weights quantization is enabled or not.

        """
        if self.final_weights_quantization_cfg:
            # if we have a final configuration, then we only care to check if it enables weights quantization
            return self.final_weights_quantization_cfg.get_attr_config(attr_name).enable_weights_quantization

        attr_candidates = self.get_all_weights_attr_candidates(attr_name)
        candidates_enable_quantization = [c.enable_weights_quantization for c in attr_candidates]
        if len(candidates_enable_quantization) > 0 and len(set(candidates_enable_quantization)) > 1:
            Logger.error(f"Weights attribute {attr_name} in node {self.name} has multiple quantization candidates "
                         f"configuration with incompatible values.")
        if all(candidates_enable_quantization):
            return True

        return False

    def is_configurable_weight(self, attr_name: str) -> bool:
        """
        Checks whether the specific weight attribute has a configurable quantization.

        Args:
            attr_name: weight attribute name.

        Returns:
            Whether the weight attribute is configurable.
        """
        return self.is_weights_quantization_enabled(attr_name) and not self.is_all_weights_candidates_equal(attr_name)

    def has_any_configurable_weight(self) -> bool:
        """
        Check whether any of the node's weights is configurable.
        Returns:
            Whether any of the node's weights is configurable.
        """
        return any(self.is_configurable_weight(attr) for attr in self.weights)

    def has_configurable_activation(self) -> bool:
        """
        Checks whether the activation has a configurable quantization.

        Returns:
            Whether the activation has a configurable quantization.
        """
        return self.is_activation_quantization_enabled() and not self.is_all_activation_candidates_equal()

    def __repr__(self):
        """

        Returns: String that represents the node.

        """
        return f'{self.type.__name__}:{self.name}'

    def is_reused(self) -> bool:
        """
        Check whether the node is reused or not
        Returns:
            True if node is reused, else False
        """
        return self.reuse or self.reuse_group is not None

    def _get_weight_name(self, name: WeightAttrT) -> List[WeightAttrT]:
        """
        Get weight names that match argument name (either string weights or integer for
        positional weights).
        Args:
            name: weight name

        Returns:
            A list of weight names that match input "name"

        """
        return [k for k in self.weights.keys()
                if (isinstance(k, int) and name == k) or (isinstance(k, str) and name in k)]

    def get_weights_by_keys(self, name: WeightAttrT) -> np.ndarray:
        """
        Get a node's weight by its name.
        Args:
            name: Name of the variable for a node's weight.

        Returns:
            A node's weight (by its name).
        """
        if name is None:
            return None

        res = self._get_weight_name(name)
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

        res = self._get_weight_name(name)
        if len(res) == 1:
            self.weights[res[0]] = tensor
        else:  # Add if not exist
            self.weights[name] = tensor
            self.weights_keys = list(self.weights.keys())  # update keys

    def get_weights_list(self):
        """

        Returns: A list of all non-positional weights the node holds.

        """
        return [self.weights[k] for k in self.weights.keys() if not isinstance(k, int)]

    def get_node_weights_attributes(self) -> List[str]:
        """

        Returns: A list of all weights attributes that the node holds.

        """
        return list(self.weights.keys())

    def insert_positional_weights_to_input_list(self, input_tensors: List) -> List:
        """
        Insert node's positional weights to input tensors list. The positional weights are inserted
        in the node's list of inputs according to their keys in the weights dictionary.

        Args:
            input_tensors: activation input tensors to node.
        Returns:
            Activation input tensors list with positional weights
        """
        for pos, weight in sorted((pos, weight) for pos, weight in self.weights.items()
                                  if isinstance(pos, int)):
            if pos > len(input_tensors):
                Logger.critical("The positional weight index cannot exceed the number of input tensors to the node.")  # pragma: no cover
            input_tensors.insert(pos, weight)

        return input_tensors

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
        # TODO: this method is used for tensorboard only. If we want to enable logging of other attributes memory
        #  then it needs to be modified. But, it might be better to remove this method from the BaseNode completely.
        kernel_attr = fw_info.get_kernel_op_attributes(self.type)[0]
        if kernel_attr is None:
            return 0
        q_params, f_params = self.get_num_parameters(fw_info)
        if self.final_weights_quantization_cfg is None:  # float coefficients
            memory = (f_params+q_params) * FP32_BYTES_PER_PARAMETER
        else:
            memory = ((f_params * FP32_BYTES_PER_PARAMETER) +
                      (q_params * self.final_weights_quantization_cfg.get_attr_config(kernel_attr).weights_n_bits
                       / 8))  # in bytes

        return memory

    def get_unified_weights_candidates_dict(self, fw_info) -> Dict[str, Any]:
        """
        In Mixed-Precision, a node's kernel can have multiple candidates for weights quantization configuration.
        In order to display a single view of a node (for example, for logging in TensorBoard) we need a way
        to create a single dictionary from all candidates.
        This method is aimed to build such an unified dictionary for a node.

        Args:
            fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize).

        Returns: A dictionary containing information from node's weight quantization configuration candidates.

        """
        shared_parameters = [CORRECTED_BIAS_ATTRIBUTE, WEIGHTS_NBITS_ATTRIBUTE]
        parameters_dict = dict()
        # We assume that only the kernel attribute have more than one candidate, since we only allow to
        # quantize the kernel using mixed precision
        # TODO: need to modify if we want to present a unified config for other attributes
        kernel_attr = fw_info.get_kernel_op_attributes(self.type)[0]
        if kernel_attr is None:
            # This node doesn't have a kernel attribute
            return {}

        if self.is_weights_quantization_enabled(kernel_attr):
            parameters_dict = copy.deepcopy(self.candidates_quantization_cfg[0].weights_quantization_cfg.
                                            get_attr_config(kernel_attr).__dict__)
            for shared_parameter in shared_parameters:
                if shared_parameter in parameters_dict:
                    unified_param = []
                    attr_candidates = self.get_all_weights_attr_candidates(kernel_attr)
                    for attr_candidate in attr_candidates:
                        unified_param.append(getattr(attr_candidate, shared_parameter))
                    parameters_dict[shared_parameter] = unified_param
        return parameters_dict

    def get_unified_activation_candidates_dict(self) -> Dict[str, Any]:
        """
        In Mixed-Precision, a node can have multiple candidates for activation quantization configuration.
        In order to display a single view of a node (for example, for logging in TensorBoard) we need a way
        to create a single dictionary from all candidates.
        This method is aimed to build such an unified dictionary for a node.

        Returns: A dictionary containing information from node's activation quantization configuration candidates.

        """
        shared_attributes = [ACTIVATION_N_BITS_ATTRIBUTE]
        attr = dict()
        if self.is_activation_quantization_enabled():
            attr = copy.deepcopy(self.candidates_quantization_cfg[0].activation_quantization_cfg.__dict__)
            for shared_attr in shared_attributes:
                if shared_attr in attr:
                    unified_attr = []
                    for candidate in self.candidates_quantization_cfg:
                        unified_attr.append(getattr(candidate.activation_quantization_cfg, shared_attr))
                    attr[shared_attr] = unified_attr
        return attr

    def is_all_activation_candidates_equal(self) -> bool:
        """
        Checks whether all candidates' quantization configuration have the same activation configuration,
        using the self-implemented __eq__ method of class NodeActivationQuantizationConfig.

        Returns: True if all candidates have same activation configuration, False otherwise.

        """
        return all(candidate.activation_quantization_cfg ==
                   self.candidates_quantization_cfg[0].activation_quantization_cfg
                   for candidate in self.candidates_quantization_cfg)

    def is_all_weights_candidates_equal(self, attr: str) -> bool:
        """
        Checks whether all candidates' quantization configuration of a given weights attribute
        have the same weights configuration,
        using the self-implemented __eq__ method of class NodeWeightsQuantizationConfig.

        Args:
            attr: The attribute name to check if all its quantization configuration candidates are equal.

        Returns: True if all the weights attribute candidates have same configuration, False otherwise.

        """
        # note that if the given attribute name does not exist in the node's attributes mapping,
        # the inner method would log an exception.
        candidates = self.get_all_weights_attr_candidates(attr)
        return all(candidate == candidates[0] for candidate in candidates[1:])

    def has_kernel_weight_to_quantize(self, fw_info):
        """
        Checks whether the node has kernel attribute that need to be quantized according to the framework info.

        Args:
            fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize).

        Returns: Whether the node has weights that need to be quantized.
        """
        attrs = fw_info.get_kernel_op_attributes(self.type)
        for attr in attrs:
            if attr and self.get_weights_by_keys(attr) is not None:
                return True
        return False

    def has_any_weight_attr_to_quantize(self) -> bool:
        """
        Checks whether the node has any weights attribute that is supposed to be quantized, based on its provided
        quantization configuration candidates.

        Returns: True if the is at least one weights attribute in the node that is supposed to be quantized.

        """

        return any([self.is_weights_quantization_enabled(attr) for attr in self.get_node_weights_attributes()])

    # TODO it makes more sense to standardize the input/output shapes at node creation.
    def get_output_shapes_list(self) -> List[tuple]:
        """
        Return output shape in a standardized form as a list of tuples.

        Returns:
            A list of output shape tuples.
        """
        # shape can be tuple or list, and multiple shapes can be packed in list or tuple
        if self.output_shape and isinstance(self.output_shape[0], (tuple, list)):
            output_shapes = [tuple(s) for s in self.output_shape]
        else:
            output_shapes = [tuple(self.output_shape)]
        return output_shapes

    def get_total_output_params(self) -> float:
        """
        Calculates the output size of the node.

        Returns: Output size.
        """
        output_shapes = self.get_output_shapes_list()
        # remove batch size (first element) from output shape
        output_shapes = [s[1:] for s in output_shapes]
        # for scalar shape (None,) prod returns 1
        return sum([np.prod([x for x in output_shape if x is not None]) for output_shape in output_shapes])

    def find_min_candidates_indices(self) -> List[int]:
        """
        Returns a list with potential minimal candidates.
        A potential minimal candidate is a candidate which its weights_n_bits and activation_n_bits pair is
        on the Pareto Front, i.e., there is no other candidate that its n_bits pair exceeds in both entries.

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

    def get_unique_weights_candidates(self, attr: str) -> List[Any]:
        """
        Returns a list with node's candidates of unique weights bit-width value for the given attribute.
        If the node have multiple candidates with the same weights bit-width for this attribute,
        the first candidate in the list is returned.

        Args:
            attr: A weights attribute name to get its unique candidates list.

        Returns: A list with node's candidates of unique weights bit-width value for the given attribute.
        """

        if attr is None or len(self.get_all_weights_attr_candidates(attr)) == 0:
            Logger.warning(f"Trying to retrieve quantization configuration candidates for attribute '{attr}', "
                           f"but such attribute can't be found in node {self.name}."
                           f"An empty list of candidates is returned.")
            return []

        unique_candidates = copy.deepcopy(self.candidates_quantization_cfg)
        seen_candidates = set()
        unique_candidates = [candidate for candidate in unique_candidates if
                             candidate.weights_quantization_cfg.get_attr_config(attr) not in seen_candidates
                             and not seen_candidates.add(candidate.weights_quantization_cfg.get_attr_config(attr))]
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

    def has_activation_quantization_enabled_candidate(self) -> bool:
        """
        Checks whether the node has quantization configuration candidates that enable activation quantization.

        Returns: True if the node has at list one quantization configuration candidate with activation quantization enabled.
        """

        return len(self.candidates_quantization_cfg) > 0 and \
            any([c.activation_quantization_cfg.enable_activation_quantization for c in self.candidates_quantization_cfg])

    def get_all_weights_attr_candidates(self, attr: str) -> List[WeightsAttrQuantizationConfig]:
        """
        Returns all WeightsAttrQuantizationConfig configuration of the given attribute of the node.

        Args:
            attr: The attribute name to get its configurations.

        Returns: A list of the attribute's quantization configurations.

        """
        # note that if the given attribute name does not exist in the node's attributes mapping,
        # the inner method would log an exception.
        return [c.weights_quantization_cfg.get_attr_config(attr) for c in self.candidates_quantization_cfg]

    def get_qco(self, fqc: FrameworkQuantizationCapabilities) -> QuantizationConfigOptions:
        """
        Get the QuantizationConfigOptions of the node according
        to the mappings from layers/LayerFilterParams to the OperatorsSet in the TargetPlatformCapabilities.

        Args:
            fqc: FQC to extract the QuantizationConfigOptions for the node.

        Returns:
            QuantizationConfigOptions of the node.
        """

        if fqc is None:
            Logger.critical(f'Can not retrieve QC options for None FQC')  # pragma: no cover

        for fl, qco in fqc.filterlayer2qco.items():
            if self.is_match_filter_params(fl):
                return qco
        # Extract qco with is_match_type to overcome mismatch of function types in TF 2.15
        matching_qcos = [_qco for _type, _qco in fqc.layer2qco.items() if self.is_match_type(_type)]
        if matching_qcos:
            if all([_qco == matching_qcos[0] for _qco in matching_qcos]):
                return matching_qcos[0]
            else:
                Logger.critical(f"Found duplicate qco types for node '{self.name}' of type '{self.type}'!")  # pragma: no cover
        return fqc.tpc.default_qco

    def filter_node_qco_by_graph(self, fqc: FrameworkQuantizationCapabilities,
                                 next_nodes: List, node_qc_options: QuantizationConfigOptions
                                 ) -> Tuple[OpQuantizationConfig, List[OpQuantizationConfig]]:
        """
        Filter quantization config options that don't match the graph.
        A node may have several quantization config options with 'activation_n_bits' values, and
        the next nodes in the graph may support different bit-width as input activation. This function
        filters out quantization config that don't comply to these attributes.

        Args:
            fqc: FQC to extract the QuantizationConfigOptions for the next nodes.
            next_nodes: Output nodes of current node.
            node_qc_options: Node's QuantizationConfigOptions.

        Returns:

        """
        # Filter quantization config options that don't match the graph.
        _base_config = node_qc_options.base_config
        _node_qc_options = node_qc_options.quantization_configurations
        if len(next_nodes):
            next_nodes_qc_options = [_node.get_qco(fqc) for _node in next_nodes]
            next_nodes_supported_input_bitwidth = min([max_input_activation_n_bits(op_cfg)
                                                       for qc_opts in next_nodes_qc_options
                                                       for op_cfg in qc_opts.quantization_configurations])

            # Filter node's QC options that match next nodes input bit-width.
            _node_qc_options = [_option for _option in _node_qc_options
                                if _option.activation_n_bits <= next_nodes_supported_input_bitwidth]
            if len(_node_qc_options) == 0:
                Logger.critical(f"Graph doesn't match FQC bit configurations: {self} -> {next_nodes}.")  # pragma: no cover

            # Verify base config match
            if any([node_qc_options.base_config.activation_n_bits > max_input_activation_n_bits(qc_opt.base_config)
                    for qc_opt in next_nodes_qc_options]):
                # base_config activation bits doesn't match next node supported input bit-width -> replace with
                # a qco from quantization_configurations with maximum activation bit-width.
                if len(_node_qc_options) > 0:
                    output_act_bitwidth = {qco.activation_n_bits: i for i, qco in enumerate(_node_qc_options)}
                    _base_config = _node_qc_options[output_act_bitwidth[max(output_act_bitwidth)]]
                    Logger.warning(f"Node {self} base quantization config changed to match Graph and FQC configuration.\nCause: {self} -> {next_nodes}.")
                else:
                    Logger.critical(f"Graph doesn't match FQC bit configurations: {self} -> {next_nodes}.")  # pragma: no cover

        return _base_config, _node_qc_options

    def is_match_type(self, _type: Type) -> bool:
        """
        Check if input type matches the node type, either in instance type or in type name.

        Args:
            _type: other node type
        Returns:
            Whether _type matches the self node type

        """
        return _type == self.type

    def is_match_filter_params(self, layer_filter_params: LayerFilterParams) -> bool:
        """
        Check if the node matches a LayerFilterParams according to its
        layer, conditions and keyword-arguments.

        Args:
            layer_filter_params: LayerFilterParams to check if the node matches its properties.

        Returns:
            Whether the node matches to the LayerFilterParams properties.
        """
        # check if provided argument is of type LayerFilterParams
        if not isinstance(layer_filter_params, LayerFilterParams):
            return False

        # Check the node has the same type as the layer in LayerFilterParams
        if not self.is_match_type(layer_filter_params.layer):
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

    def get_simd(self) -> int:
        """
        Retrieves the SIMD size used for this node. It collects the SIMD sizes from all candidate
        configurations and returns the minimum SIMD size.

        Returns:
            int: The node's SIMD size.

        """
        simd_list = [qc.weights_quantization_cfg.simd_size for qc in self.candidates_quantization_cfg]
        if len(simd_list) > 1:
            Logger.warning(f"More than one pruning SIMD option is available."
                           f" Min SIMD is used: {min(simd_list)}")
        if len(simd_list) == 0:  # pragma: no cover
            Logger.critical(f"No SIMD option is available for {self}")
        _simd = min(simd_list)
        if _simd <= 0 or int(_simd) != _simd:  # pragma: no cover
            Logger.critical(f"SIMD is expected to be a non-positive integer but found: {_simd}")
        return _simd

    def sort_node_candidates(self, fw_info):
        """
        Sorts the node candidates.
        We assume that the candidates are ordered in the following way (for mixed precision purposes):
            - If the node has a kernel attribute, then we use the kernel weights number of bits to sort the candidates
            (in descending order). We use the candidate activation number of bits as a secondary order.
            - If the node doesn't have a kernel we only consider the candidate activation number of bits to sort
            the candidates in descending order.
        The operation is done inplace.

        Args:
            fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize).

        """
        if self.candidates_quantization_cfg is not None:
            kernel_attr = fw_info.get_kernel_op_attributes(self.type)[0]
            if kernel_attr is not None:
                self.candidates_quantization_cfg.sort(
                    key=lambda c: (c.weights_quantization_cfg.get_attr_config(kernel_attr).weights_n_bits,
                                   c.activation_quantization_cfg.activation_n_bits), reverse=True)
            else:
                self.candidates_quantization_cfg.sort(key=lambda c: c.activation_quantization_cfg.activation_n_bits,
                                                      reverse=True)
