# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
from unittest.mock import Mock

import abc
from typing import Dict, Optional, Type, Callable, Tuple

import numpy as np
import pytest

from model_compression_toolkit.core import MixedPrecisionQuantizationConfig, CoreConfig, QuantizationConfig, \
    FrameworkInfo
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.mixed_precision.sensitivity_eval.metric_calculators import \
    DistanceMetricCalculator, CustomMetricCalculator
from model_compression_toolkit.core.common.mixed_precision.sensitivity_eval.sensitivity_evaluation import SensitivityEvaluation
from model_compression_toolkit.core.quantization_prep_runner import quantization_preparation_runner
from model_compression_toolkit.target_platform_capabilities import AttributeQuantizationConfig, OpQuantizationConfig, \
    Signedness, QuantizationConfigOptions, OperatorSetNames, TargetPlatformCapabilities, QuantizationMethod
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR
from tests_pytest._test_util.fw_test_base import BaseFWIntegrationTest
from tests_pytest._test_util.tpc_util import configure_mp_opsets_for_kernel_bias_ops, configure_mp_activation_opsets


def build_tpc(w_nbits=(8, 4, 2), a_nbits=(16, 8)):
    default_w_cfg = AttributeQuantizationConfig(weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
                                                weights_n_bits=8,
                                                weights_per_channel_threshold=False,
                                                enable_weights_quantization=True)
    default_op_cfg = OpQuantizationConfig(
        default_weight_attr_config=default_w_cfg.clone_and_edit(enable_weights_quantization=False),
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=[16, 8],
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None, fixed_zero_point=None, simd_size=32, signedness=Signedness.AUTO)

    # make bias quantizable
    default_w_op_cfg = default_op_cfg.clone_and_edit(
        attr_weights_configs_mapping={KERNEL_ATTR: default_w_cfg, BIAS_ATTR: default_w_cfg}
    )
    default_cfg = QuantizationConfigOptions(quantization_configurations=[default_op_cfg])

    # configurable activation + weights
    ops_conv, _ = configure_mp_opsets_for_kernel_bias_ops(opset_names=[OperatorSetNames.CONV],
                                                          base_w_config=default_w_cfg, base_op_config=default_w_op_cfg,
                                                          w_nbits=w_nbits, a_nbits=a_nbits)
    # configurable activations
    ops_relu, _ = configure_mp_activation_opsets(opset_names=[OperatorSetNames.RELU],
                                                 base_op_config=default_op_cfg, a_nbits=a_nbits)
    # configurable activation
    ops_convtr, _ = configure_mp_opsets_for_kernel_bias_ops(opset_names=[OperatorSetNames.CONV_TRANSPOSE],
                                                            base_w_config=default_w_cfg, base_op_config=default_w_op_cfg,
                                                            w_nbits=[default_w_cfg.weights_n_bits], a_nbits=a_nbits)

    # configurable weights
    ops_fc, _ = configure_mp_opsets_for_kernel_bias_ops(opset_names=[OperatorSetNames.FULLY_CONNECTED],
                                                        base_w_config=default_w_cfg, base_op_config=default_w_op_cfg,
                                                        w_nbits=w_nbits, a_nbits=[default_op_cfg.activation_n_bits])

    tpc = TargetPlatformCapabilities(default_qco=default_cfg, tpc_platform_type='test',
                                     operator_set=ops_conv+ops_convtr+ops_fc+ops_relu, fusing_patterns=None)
    return tpc


class BaseSensitivityEvaluationIntegTester(BaseFWIntegrationTest, abc.ABC):
    """ Test quantization layers and quantizers configuration when building the models and configuring mp model. """
    BIAS: str
    KERNEL: str
    conv_cls: Type
    relu_cls: Type
    convtr_cls: Type
    fc_cls: Type

    # these will come from FwMixin
    fw_info: FrameworkInfo
    fw_impl: FrameworkImplementation
    fetch_model_layers_by_cls: Callable

    input_shape = (1, 3, 16, 16)

    @abc.abstractmethod
    def build_model(self, input_shape):
        """ Build fw model: conv -> relu -> convtr -> flatten -> fc. """
        pass

    @abc.abstractmethod
    def infer_models(self, orig_model, mp_model, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Run inference for input model and mp model on a given input. """
        pass

    def repr_datagen(self):
        yield [np.random.rand(*self.input_shape)]

    def _setup(self, mp_config=None):
        model = self.build_model(self.input_shape)
        tpc = build_tpc(w_nbits=(8, 4, 2), a_nbits=(16, 8))
        mp_config = mp_config or MixedPrecisionQuantizationConfig()
        core_config = CoreConfig(quantization_config=QuantizationConfig(),
                                 mixed_precision_config=mp_config)
        g = self.run_graph_preparation(model, self.repr_datagen, tpc=tpc,
                                       quant_config=core_config.quantization_config, mp=True)
        g = quantization_preparation_runner(g, self.repr_datagen, core_config=core_config,
                                            fw_impl=self.fw_impl, hessian_info_service=None)
        se = SensitivityEvaluation(g, core_config.mixed_precision_config, self.repr_datagen,
                                   fw_impl=self.fw_impl, disable_activation_for_metric=False,
                                   hessian_info_service=None)
        return model, g, se

    def test_build_models(self):
        """ Test quant layers and quantizers are built and configured correctly for the mp model,
            and the mapping from nodes to configurable quant layers is correct. """
        model, g, se = self._setup()
        # sanity check that we set up the configuration we intended, and it was correctly configured
        conf_a_nodes = [n for n in g.get_topo_sorted_nodes() if n.has_configurable_activation()]
        assert [n.layer_class for n in conf_a_nodes] == [self.conv_cls, self.relu_cls, self.convtr_cls]
        conf_w_nodes = [n for n in g.get_topo_sorted_nodes() if n.has_any_configurable_weight()]
        assert [n.layer_class for n in conf_w_nodes] == [self.conv_cls, self.fc_cls]
        # we quantize biases, to make sure their quantization is going to be disabled since they are not configurable
        assert all(n.is_weights_quantization_enabled(self.BIAS) for n in conf_w_nodes)

        # only configurable activations / weights should have quantization layers
        # and all quantizers should be disabled. They will be turned on one by one per candidate.
        activation_holders = self.fetch_model_layers_by_cls(se.mp_model, self.fw_impl.activation_quant_layer_cls)
        assert len(activation_holders) == len(conf_a_nodes)
        for ah in activation_holders:
            assert isinstance(ah.activation_holder_quantizer, self.fw_impl.configurable_activation_quantizer_cls)
            assert ah.activation_holder_quantizer.active_quantization_config_index is None
        weight_wrappers = self.fetch_model_layers_by_cls(se.mp_model, self.fw_impl.weights_quant_layer_cls)
        for ww in weight_wrappers:
            # bias should not have quantizer
            assert len(ww.weights_quantizers) == 1
            assert isinstance(ww.weights_quantizers[self.KERNEL], self.fw_impl.configurable_weights_quantizer_cls)
            assert ww.weights_quantizers[self.KERNEL].active_quantization_config_index is None

        # verify mapping from graph nodes to configurable model layers
        assert len(se.conf_node2layers) == 4
        conv, relu, convtr = conf_a_nodes
        _, fc = conf_w_nodes
        assert len(se.conf_node2layers[conv.name]) == 2
        assert set(se.conf_node2layers[conv.name]) == {activation_holders[0], weight_wrappers[0]}
        assert len(se.conf_node2layers[relu.name]) == 1 and se.conf_node2layers[relu.name][0] == activation_holders[1]
        assert len(se.conf_node2layers[convtr.name]) == 1 and se.conf_node2layers[convtr.name][0] == activation_holders[2]
        assert len(se.conf_node2layers[fc.name]) == 1 and se.conf_node2layers[fc.name][0] == weight_wrappers[1]

        # baseline float model doesn't contain any quant layers
        activation_holders = self.fetch_model_layers_by_cls(se.metric_calculator.ref_model, self.fw_impl.activation_quant_layer_cls)
        weight_wrappers = self.fetch_model_layers_by_cls(se.metric_calculator.ref_model, self.fw_impl.weights_quant_layer_cls)
        assert not activation_holders and not weight_wrappers

        # sanity check that initial mp model is indeed un-quantized (identical to original model)
        x = next(self.repr_datagen())[0]
        y, y_mp = self.infer_models(model, se.mp_model, x)
        assert np.array_equal(y, y_mp)
        # sanity check that baseline model is identical to original model
        _, y_float = self.infer_models(model, se.metric_calculator.ref_model, x)
        assert np.array_equal(y, y_float)

    def test_build_models_disable_activations(self):
        """ Test mp model with disabled activation flag. """
        model, g, _ = self._setup()
        se = SensitivityEvaluation(g, MixedPrecisionQuantizationConfig(), self.repr_datagen,
                                   fw_impl=self.fw_impl, disable_activation_for_metric=True, hessian_info_service=None)
        activation_holders = self.fetch_model_layers_by_cls(se.mp_model, self.fw_impl.activation_quant_layer_cls)
        assert not activation_holders
        weight_wrappers = self.fetch_model_layers_by_cls(se.mp_model, self.fw_impl.weights_quant_layer_cls)
        assert len(weight_wrappers) == 2

    def test_configure_mp_model(self):
        """ Test mp model configuration. """
        model, g, se = self._setup()
        # a x w: conv 2x2, relu 2, conv_tr 2x1, fc 1x2
        conv, relu, conv_tr = [n.name for n in g.get_topo_sorted_nodes() if n.has_configurable_activation()]
        conv_, fc = [n.name for n in g.get_topo_sorted_nodes() if n.has_any_configurable_weight()]
        assert conv is conv_

        with se._configured_mp_model(mp_a_cfg={conv: 2}, mp_w_cfg={conv: 1}):
            self._validate_mpmodel_quant_layers(se, {conv: 2, relu: None, conv_tr: None}, {conv: 1, fc: None})
            # sanity test that model was indeed quantized
            y, y_mp = self.infer_models(model, se.mp_model, next(self.repr_datagen())[0])
            assert not np.allclose(y, y_mp)

        # restored to un-quantized
        self._validate_mpmodel_quant_layers(se, {conv: None, relu: None, conv_tr: None}, {conv: None, fc: None})

        with se._configured_mp_model(mp_a_cfg={conv: 1}, mp_w_cfg={}):
            self._validate_mpmodel_quant_layers(se, {conv: 1, relu: None, conv_tr: None}, {conv: None, fc: None})

        with se._configured_mp_model(mp_a_cfg={}, mp_w_cfg={conv: 3}):
            self._validate_mpmodel_quant_layers(se, {conv: None, relu: None, conv_tr: None}, {conv: 3, fc: None})

        with se._configured_mp_model(mp_a_cfg={conv: None, relu: 1, conv_tr: 0}, mp_w_cfg={fc: 2}):
            self._validate_mpmodel_quant_layers(se, {conv: None, relu: 1, conv_tr: 0}, {conv: None, fc: 2})

        # restored to un-quantized
        self._validate_mpmodel_quant_layers(se, {conv: None, relu: None, conv_tr: None}, {conv: None, fc: None})

    def test_configure_mp_model_errors(self):
        """ Test errors during model configuration. """
        model, g, se = self._setup()
        with pytest.raises(ValueError, match='Requested configuration is either empty or contain only None values'):
            with se._configured_mp_model({}, {}):
                pass

        conv, relu, conv_tr = [n.name for n in g.get_topo_sorted_nodes() if n.has_configurable_activation()]
        _, fc = [n.name for n in g.get_topo_sorted_nodes() if n.has_any_configurable_weight()]
        with pytest.raises(ValueError, match='Requested configuration is either empty or contain only None values'):
            with se._configured_mp_model({relu: None}, {conv: None}):
                pass

        with pytest.raises(ValueError, match='Not all mp configs were consumed'):
            with se._configured_mp_model({conv: 1}, {conv: 0, relu: None}):
                pass

        with pytest.raises(ValueError, match='Not all mp configs were consumed'):
            with se._configured_mp_model({conv: 1, fc: 0}, {conv: 0}):
                pass

    def _run_test_compute_metric_method(self, custom, mocker):
        """ Test compute_metric method (not the metric computation itself) """

        mp_config = MixedPrecisionQuantizationConfig(custom_metric_fn=Mock()) if custom else None
        model, g, se = self._setup(mp_config)
        conv, relu, conv_tr = [n.name for n in g.get_topo_sorted_nodes() if n.has_configurable_activation()]
        conv_, fc = [n.name for n in g.get_topo_sorted_nodes() if n.has_any_configurable_weight()]

        def mock_compute_metric(*args):
            # validate correct configuration inside compute metric
            self._validate_mpmodel_quant_layers(se, {conv: 2, relu: 0, conv_tr: None}, {conv: None, fc: 1})
            return 5

        calc_cls = CustomMetricCalculator if custom else DistanceMetricCalculator
        mocker.patch.object(calc_cls, 'compute', mock_compute_metric)

        # initial model is un-quantized
        self._validate_mpmodel_quant_layers(se, {conv: None, relu: None, conv_tr: None}, {conv: None, fc: None})
        mp_a_cfg = {conv: 2, relu: 0}
        mp_w_cfg = {fc: 1}

        res = se.compute_metric(mp_a_cfg, mp_w_cfg)
        assert res == 5
        # restored to un-quantized
        self._validate_mpmodel_quant_layers(se, {conv: None, relu: None, conv_tr: None}, {conv: None, fc: None})

    def _validate_mpmodel_quant_layers(self, se: SensitivityEvaluation, exp_node_a_indices: Dict[str, Optional[int]],
                                       exp_node_w_indices: Dict[str, Optional[int]]):
        self._validate_mpmodel_a_quant_layers(se, exp_node_a_indices)
        self._validate_mpmodel_w_quant_layers(se, exp_node_w_indices)

    def _validate_mpmodel_a_quant_layers(self, se: SensitivityEvaluation, exp_node_indices: Dict[str, Optional[int]]):
        exp_a_layer_indicies = {}
        for n, ind in exp_node_indices.items():
            layers = [ql for ql in se.conf_node2layers[n] if isinstance(ql, self.fw_impl.activation_quant_layer_cls)]
            assert len(layers) == 1
            exp_a_layer_indicies[layers[0]] = ind

        activation_holders = self.fetch_model_layers_by_cls(se.mp_model, self.fw_impl.activation_quant_layer_cls)
        for ah in activation_holders:
            assert ah.activation_holder_quantizer.active_quantization_config_index == exp_a_layer_indicies[ah]

    def _validate_mpmodel_w_quant_layers(self, se: SensitivityEvaluation, exp_node_indices: Dict[str, Optional[int]]):
        exp_w_layer_indicies = {}
        for n, ind in exp_node_indices.items():
            layers = [ql for ql in se.conf_node2layers[n] if isinstance(ql, self.fw_impl.weights_quant_layer_cls)]
            assert len(layers) == 1
            exp_w_layer_indicies[layers[0]] = ind

        weight_wrappers = self.fetch_model_layers_by_cls(se.mp_model, self.fw_impl.configurable_weights_quantizer_cls)
        for ww in weight_wrappers:
            assert ww.weights_quantizers[self.KERNEL].active_quantization_config_index == exp_w_layer_indicies[ww]
