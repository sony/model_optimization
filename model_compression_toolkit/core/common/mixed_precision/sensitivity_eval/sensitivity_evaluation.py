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
import contextlib
import copy
import itertools

from typing import Callable, Any, Tuple, Dict, Optional

from model_compression_toolkit.core import FrameworkInfo, MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.mixed_precision.sensitivity_eval.metric_calculators import \
    CustomMetricCalculator, DistanceMetricCalculator
from model_compression_toolkit.core.common.mixed_precision.sensitivity_eval.set_layer_to_bitwidth import \
    set_activation_quant_layer_to_bitwidth, set_weights_quant_layer_to_bitwidth
from model_compression_toolkit.core.common.quantization.node_quantization_config import ActivationQuantizationMode
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.common.hessian import HessianInfoService


class SensitivityEvaluation:
    """
    Sensitivity evaluation of a bit-width configuration for Mixed Precision search.
    """

    def __init__(self,
                 graph: Graph,
                 mp_config: MixedPrecisionQuantizationConfig,
                 representative_data_gen: Callable,
                 fw_impl: Any,
                 disable_activation_for_metric: bool = False,
                 hessian_info_service: HessianInfoService = None
                 ):
        """
        Args:
            graph: Graph to search for its MP configuration.
            mp_config: MP Quantization configuration for how the graph should be quantized.
            representative_data_gen: Dataset used for getting batches for inference.
            fw_impl: FrameworkImplementation object with a specific framework methods implementation.
            disable_activation_for_metric: Whether to disable activation quantization when computing the MP metric.
            hessian_info_service: HessianInfoService to fetch Hessian approximation information.

        """
        self.mp_config = mp_config
        self.representative_data_gen = representative_data_gen
        self.fw_impl = fw_impl

        if self.mp_config.custom_metric_fn:
            self.metric_calculator = CustomMetricCalculator(graph, self.mp_config.custom_metric_fn)
        else:
            self.metric_calculator = DistanceMetricCalculator(graph, mp_config, representative_data_gen,
                                                              fw_impl=fw_impl,
                                                              hessian_info_service=hessian_info_service)

        # Build a mixed-precision model which can be configured to use different bitwidth in different layers.
        # Also, returns a mapping between a configurable graph's node and its matching layer(s) in the built MP model.
        self.mp_model, self.conf_node2layers = self._build_mp_model(graph, self.metric_calculator.all_interest_points,
                                                                    disable_activation_for_metric)

    def compute_metric(self, mp_a_cfg: Dict[str, Optional[int]], mp_w_cfg: Dict[str, Optional[int]]) -> float:
        """
        Compute the sensitivity metric of the MP model for a given configuration.
        Quantization for any configurable activation / weight that were not passed is disabled.

        Args:
            mp_a_cfg: Bitwidth activations configuration for the MP model.
            mp_w_cfg: Bitwidth weights configuration for the MP model.

        Returns:
            The sensitivity metric of the MP model for a given configuration.
        """
        with self._configured_mp_model(mp_a_cfg, mp_w_cfg):
            sensitivity_metric = self.metric_calculator.compute(self.mp_model)

        return sensitivity_metric

    def _build_mp_model(self, graph, outputs, disable_activations: bool) -> Tuple[Any, dict]:
        """
        Builds an MP model with configurable layers.

        Returns:
            MP model and a mapping from configurable graph nodes to their corresponding quantization layer(s)
            in the MP model.
        """
        evaluation_graph = copy.deepcopy(graph)

        # Disable quantization for non-configurable nodes, and, if requested, for all activations (quantizers won't
        # be added to the model).
        for n in evaluation_graph.get_topo_sorted_nodes():
            if disable_activations or not n.has_configurable_activation():
                for c in n.candidates_quantization_cfg:
                    c.activation_quantization_cfg.quant_mode = ActivationQuantizationMode.NO_QUANT
            if not n.has_any_configurable_weight():
                for c in n.candidates_quantization_cfg:
                    c.weights_quantization_cfg.disable_all_weights_quantization()

        model_mp, _, conf_node2layers = self.fw_impl.model_builder(evaluation_graph,
                                                                   mode=ModelBuilderMode.MIXEDPRECISION,
                                                                   append2output=outputs)

        # Disable all configurable quantizers. They will be activated one at a time during sensitivity evaluation.
        for layer in itertools.chain(*conf_node2layers.values()):
            if isinstance(layer, self.fw_impl.activation_quant_layer_cls):
                set_activation_quant_layer_to_bitwidth(layer, None, self.fw_impl)
            else:
                assert isinstance(layer, self.fw_impl.weights_quant_layer_cls)
                set_weights_quant_layer_to_bitwidth(layer, None, self.fw_impl)

        return model_mp, conf_node2layers

    @contextlib.contextmanager
    def _configured_mp_model(self, mp_a_cfg: Dict[str, Optional[int]], mp_w_cfg: Dict[str, Optional[int]]):
        """
        Context manager to configure specific configurable layers of the mp model. At exit, configuration is
        automatically restored to un-quantized.

        Args:
            mp_a_cfg: Nodes bitwidth indices to configure activation quantizers to.
            mp_w_cfg: Nodes bitwidth indices to configure weights quantizers to.

        """
        if not (mp_a_cfg and any(v is not None for v in mp_a_cfg.values()) or
                mp_w_cfg and any(v is not None for v in mp_w_cfg.values())):
            raise ValueError(f'Requested configuration is either empty or contain only None values.')

        # defined here so that it can't be used directly
        def apply_bitwidth_config(a_cfg, w_cfg):
            node_names = set(a_cfg.keys()).union(set(w_cfg.keys()))
            for n in node_names:
                node_quant_layers = self.conf_node2layers.get(n)
                if node_quant_layers is None:    # pragma: no cover
                    raise ValueError(f"Matching layers for node {n} not found in the mixed precision model configuration.")
                for qlayer in node_quant_layers:
                    assert isinstance(qlayer, (self.fw_impl.activation_quant_layer_cls,
                                               self.fw_impl.weights_quant_layer_cls)), f'Unexpected {type(qlayer)} of node {n}'
                    if isinstance(qlayer, self.fw_impl.activation_quant_layer_cls) and n in a_cfg:
                        set_activation_quant_layer_to_bitwidth(qlayer, a_cfg[n], self.fw_impl)
                        a_cfg.pop(n)
                    elif isinstance(qlayer, self.fw_impl.weights_quant_layer_cls) and n in w_cfg:
                        set_weights_quant_layer_to_bitwidth(qlayer, w_cfg[n], self.fw_impl)
                        w_cfg.pop(n)
            if a_cfg or w_cfg:
                raise ValueError(f'Not all mp configs were consumed, remaining activation config {a_cfg}, '
                                 f'weights config {w_cfg}.')

        apply_bitwidth_config(mp_a_cfg.copy(), mp_w_cfg.copy())
        try:
            yield
        finally:
            apply_bitwidth_config({n: None for n in mp_a_cfg}, {n: None for n in mp_w_cfg})


