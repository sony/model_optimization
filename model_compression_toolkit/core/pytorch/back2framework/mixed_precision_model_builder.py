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

from typing import List, Any, Tuple, Union, Dict

import torch
from mct_quantizers import PytorchQuantizationWrapper, QuantizationTarget, \
    PytorchActivationQuantizationHolder
from mct_quantizers.common.constants import ACTIVATION_HOLDER_QUANTIZER
from mct_quantizers.common.get_quantizers import get_inferable_quantizer_class
from mct_quantizers.pytorch.quantizers import BasePyTorchInferableQuantizer

from model_compression_toolkit.core import FrameworkInfo
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PyTorchModelBuilder

from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.mixed_precision.configurable_activation_quantizer import \
    ConfigurableActivationQuantizer
from model_compression_toolkit.core.pytorch.mixed_precision.configurable_weights_quantizer import \
    ConfigurableWeightsQuantizer

from model_compression_toolkit.exporter.model_wrapper.pytorch.builder.node_to_quantizer import \
    get_weights_inferable_quantizer_kwargs, get_activation_inferable_quantizer_kwargs
from model_compression_toolkit.logger import Logger


class MixedPrecisionPyTorchModelBuilder(PyTorchModelBuilder):
    """
    Mixed-precision PyTorch model.
    """
    def __init__(self,
                 graph: common.Graph,
                 append2output=None,
                 fw_info: FrameworkInfo = DEFAULT_PYTORCH_INFO,
                 return_float_outputs: bool = False):
        """

        Args:
            graph: Graph to build the model from.
            append2output: Nodes to append to model's output.
            fw_info: Information about the specific framework of the model that is built.
            return_float_outputs: Whether the model returns float tensors or not.
        """

        self.graph = graph

        super().__init__(graph,
                         append2output,
                         fw_info,
                         return_float_outputs,
                         wrapper=self.mixed_precision_wrapper,
                         get_activation_quantizer_holder_fn=self.mixed_precision_activation_holder)

    def mixed_precision_wrapper(self,
                                n: common.BaseNode,
                                layer: torch.nn.Module) -> Union[PytorchQuantizationWrapper, torch.nn.Module]:
        """
        A function which takes a computational graph node and a pytorch layer and perform the quantization
        wrapping for mixed precision.

        Args:
            n: A node of mct graph.
            layer: A pytorch layer

        Returns: Wrapped layer with a configurable quantizer if the layer should quantized in mixed precision,
        otherwise returns either the layer wrapped with a fixed precision inferable quantizer or the layer as is if it's
        not supposed to be quantized.

        """

        weights_conf_nodes_names = [n.name for n in self.graph.get_weights_configurable_nodes()]

        if n.is_weights_quantization_enabled():
            kernel_attributes = self.fw_info.get_kernel_op_attributes(n.type)
            if n.name in weights_conf_nodes_names:
                return PytorchQuantizationWrapper(layer,
                                                  weights_quantizers={attr: ConfigurableWeightsQuantizer(
                                                      **self._get_weights_configurable_quantizer_kwargs(n, attr))
                                                      for attr in kernel_attributes})
            else:
                node_weights_qc = n.get_unique_weights_candidates()
                if not len(node_weights_qc) == 1:
                    Logger.error(f"Expecting node {n.name} to have a unique weights configuration "  # pragma: no cover
                                 f"but {len(node_weights_qc)} different configurations exist.")

                quantier_for_node = get_inferable_quantizer_class(QuantizationTarget.Weights,
                                                                  node_weights_qc[0].weights_quantization_cfg.weights_quantization_method,
                                                                  BasePyTorchInferableQuantizer)
                kwargs = get_weights_inferable_quantizer_kwargs(node_weights_qc[0].weights_quantization_cfg)

                return PytorchQuantizationWrapper(layer,
                                                  weights_quantizers={attr: quantier_for_node(**kwargs)
                                                                      for attr in kernel_attributes})

        return layer

    def _get_weights_configurable_quantizer_kwargs(self, n: BaseNode, attr: str) -> Dict[str, Any]:
        """
        Get the quantization parameters for a configurable quantizer.

        Args:
            n: The node for which the quantizer is being created.
            attr: The name of the weights attribute to be quantized.

        Returns:
            The quantization parameters as a dictionary.
        """

        assert n.candidates_quantization_cfg is not None, f"Node {n.name} candidates_quantization_cfg is None"
        node_q_cfg_candidates = n.candidates_quantization_cfg
        # sort by descending bit width so using indices would be easier
        node_q_cfg_candidates.sort(key=lambda x: (x.weights_quantization_cfg.weights_n_bits,
                                                  x.activation_quantization_cfg.activation_n_bits), reverse=True)

        float_weights = n.get_weights_by_keys(attr)

        max_cfg_candidates = n.find_max_candidates_indices()
        if not len(max_cfg_candidates) == 1:
            Logger.error(f"A maximal config candidate must be defined, "  # pragma: no cover
                         f"but some node have multiple potential maximal candidates")

        max_candidate_idx = max_cfg_candidates[0]

        return {'node_q_cfg': node_q_cfg_candidates,
                'float_weights': float_weights,
                'max_candidate_idx': max_candidate_idx
                }

    def mixed_precision_activation_holder(self, n: BaseNode) -> PytorchActivationQuantizationHolder:
        """
        Retrieve a PytorchActivationQuantizationHolder layer to use for activation quantization for a node.
        The layer should hold either a configurable activation quantizer, if it is quantized with mixed precision,
        or an inferable quantizer for fixed single bit-width quantization.

        Args:
            n: Node to get PytorchActivationQuantizationHolder to attach in its output.

        Returns:
            A PytorchActivationQuantizationHolder layer for the node activation quantization.
        """

        activation_conf_nodes_names = [n.name for n in self.graph.get_activation_configurable_nodes()]

        activation_quantizers = []
        if n.is_activation_quantization_enabled():
            num_of_outputs = len(n.output_shape) if isinstance(n.output_shape, list) else 1
            if n.name in activation_conf_nodes_names:
                assert n.candidates_quantization_cfg is not None, f"Node {n.name} candidates_quantization_cfg is None"
                node_q_cfg_candidates = n.candidates_quantization_cfg
                # sort by descending bit width so using indices would be easier
                node_q_cfg_candidates.sort(key=lambda x: (x.weights_quantization_cfg.weights_n_bits,
                                                          x.activation_quantization_cfg.activation_n_bits),
                                           reverse=True)

                max_cfg_candidates = n.find_max_candidates_indices()
                assert len(max_cfg_candidates) == 1, \
                    f"A maximal config candidate must be defined, but some node have multiple potential maximal candidates"
                max_candidate_idx = max_cfg_candidates[0]

                activation_quantizers = [ConfigurableActivationQuantizer(**{'node_q_cfg': node_q_cfg_candidates,
                                                                            'max_candidate_idx': max_candidate_idx})] \
                                        * num_of_outputs
            else:
                node_act_qc = n.get_unique_activation_candidates()
                assert len(node_act_qc) == 1, f"Expecting node {n.name} to have a unique activation configuration, " \
                                              f"but {len(node_act_qc)} different configurations exist."
                quantizer_for_node = get_inferable_quantizer_class(QuantizationTarget.Activation,
                                                                   node_act_qc[0].activation_quantization_cfg.activation_quantization_method,
                                                                   BasePyTorchInferableQuantizer)
                kwargs = get_activation_inferable_quantizer_kwargs(node_act_qc[0].activation_quantization_cfg)

                activation_quantizers = [quantizer_for_node(**kwargs)] * num_of_outputs

        # Holder by definition uses a single quantizer for the activation quantization
        # thus we make sure this is the only possible case (unless it's a node with no activation
        # quantization, which in this case has an empty list).
        if len(activation_quantizers) == 1:
            return PytorchActivationQuantizationHolder(activation_quantizers[0])

        Logger.error(f'PytorchActivationQuantizationHolder supports a single quantizer but '  # pragma: no cover
                     f'{len(activation_quantizers)} quantizers were found for node {n}')

    def build_model(self) -> Tuple[torch.nn.Module, UserInformation,
                                   Dict[str, Union[PytorchQuantizationWrapper, PytorchActivationQuantizationHolder]]]:
        """
        Build a PyTorch float model and return it.
        Returns: Float PyTorch model and user information.

        """
        model, user_info = super().build_model()

        # creating a mapping between graph nodes and model's layers for mixed precision configurability
        model_layers = dict(model.named_children())
        conf_node2layers = {n.name: self._find_layers_in_model_by_node(n, model_layers)
                            for n in self.graph.get_configurable_sorted_nodes()}

        return model, user_info, conf_node2layers


    @staticmethod
    def _get_weights_quant_layers(n: BaseNode, named_layers: Dict[str, torch.nn.Module]) \
            -> List[PytorchQuantizationWrapper]:
        """
        Filters PytorchQuantizationWrapper layers from an MP model that are matching to the given graph node.

        Args:
            n: A configurable graph node.
            named_layers: Mixed precision model layers list.

        Returns: A list of layers that responsible for the node's weights quantization.

        """
        return [module for m_name, module in named_layers.items() if isinstance(module, PytorchQuantizationWrapper)
                and m_name == n.name]

    @staticmethod
    def _get_activation_quant_layers(n: BaseNode, named_layers: Dict[str, torch.nn.Module]) \
            -> List[PytorchActivationQuantizationHolder]:
        """
        Filters PytorchActivationQuantizationHolder layers from an MP model that are matching to the given graph node.

        Args:
            n: A configurable graph node.
            named_layers: Mixed precision model layers list.

        Returns: A list of layers that responsible for the node's activation quantization.

        """
        return [module for m_name, module in named_layers.items()
                if isinstance(module, PytorchActivationQuantizationHolder) and
                m_name.replace(f"_{ACTIVATION_HOLDER_QUANTIZER}", '') == n.name]

    def _find_layers_in_model_by_node(self, n: BaseNode, named_layers: Dict[str, torch.nn.Module]) -> \
            List[Union[PytorchQuantizationWrapper, PytorchActivationQuantizationHolder]]:
        """
        Retries layers from an MP model that are matching to the given graph node, that is, this are either
        PytorchQuantizationWrapper layers or PytorchActivationQuantizationHolder layers that are responsible for the graph
        configurable model quantization.

        Args:
            n: A configurable graph node.
            named_layers: Mixed precision model layers list.

        Returns: A list of layers that responsible for the node's quantization.

        """
        weights_quant = n.is_weights_quantization_enabled()
        act_quant = n.is_activation_quantization_enabled()

        if weights_quant and not act_quant:
            return self._get_weights_quant_layers(n, named_layers)
        elif not weights_quant and act_quant:
            return self._get_activation_quant_layers(n, named_layers)
        elif weights_quant and act_quant:
            return self._get_weights_quant_layers(n, named_layers) + self._get_activation_quant_layers(n, named_layers)
        else:  # pragma: no cover
            Logger.error(f"Expects node {n.name} to have at either weights or activation quantization configured,"
                         f"but both are disabled.")
