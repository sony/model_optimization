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

import tensorflow as tf
from tqdm import tqdm

from model_compression_toolkit.core import DebugConfig
from model_compression_toolkit.core.common.mixed_precision.bit_width_setter import set_bit_widths
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_facade import search_bit_width
from model_compression_toolkit.core.common.model_collector import ModelCollector
from model_compression_toolkit.core.common.network_editors.actions import EditRule, ChangeFinalWeightsQuantConfigAttr, \
    ChangeCandidatesActivationQuantConfigAttr, \
    ChangeCandidatesWeightsQuantConfigAttr, ChangeCandidatesActivationQuantizationMethod, \
    ChangeCandidatesWeightsQuantizationMethod, ChangeFinalWeightsQuantizationMethod, \
    ChangeFinalActivationQuantizationMethod
from model_compression_toolkit.core.common.network_editors.edit_network import edit_network_graph
from model_compression_toolkit.core.common.network_editors.node_filters import NodeTypeFilter
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_computation import \
    calculate_quantization_params
from model_compression_toolkit.core.common.statistics_correction.statistics_correction import \
    statistics_correction_runner
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner
from model_compression_toolkit.core.keras.constants import KERNEL
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod


keras = tf.keras
layers = keras.layers


def prepare_graph_for_first_network_editor(in_model, representative_data_gen, core_config, fw_info, fw_impl,
                                           tpc, target_resource_utilization=None, tb_w=None):

    if target_resource_utilization is not None:
        core_config.mixed_precision_config.set_mixed_precision_enable()

    transformed_graph = graph_preparation_runner(in_model,
                                                 representative_data_gen,
                                                 core_config.quantization_config,
                                                 fw_info,
                                                 fw_impl,
                                                 tpc,
                                                 core_config.bit_width_config,
                                                 tb_w,
                                                 mixed_precision_enable=core_config.is_mixed_precision_enabled)


    ######################################
    # Statistic collection
    ######################################
    mi = ModelCollector(transformed_graph,
                        fw_impl,
                        fw_info,
                        core_config.quantization_config)

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'after_statistic_collection')

    for _data in tqdm(representative_data_gen()):
        mi.infer(_data)

    ######################################
    # Edit network according to user
    # specific settings
    ######################################
    # Notice that not all actions affect at this stage (for example, actions that edit the final configuration as
    # there are no final configurations at this stage of the optimization). For this reason we edit the graph
    # again at the end of the optimization process.
    edit_network_graph(transformed_graph, fw_info, core_config.debug_config.network_editor)

    return transformed_graph


def prepare_graph_for_second_network_editor(in_model, representative_data_gen, core_config, fw_info, fw_impl,
                                            tpc, target_resource_utilization=None, tb_w=None):
    transformed_graph = prepare_graph_for_first_network_editor(in_model=in_model,
                                                               representative_data_gen=representative_data_gen,
                                                               core_config=core_config,
                                                               fw_info=fw_info,
                                                               fw_impl=fw_impl,
                                                               tpc=tpc,
                                                               target_resource_utilization=target_resource_utilization,
                                                               tb_w=tb_w)

    if target_resource_utilization is not None:
        core_config.mixed_precision_config.set_mixed_precision_enable()

    ######################################
    # Calculate quantization params
    ######################################
    calculate_quantization_params(transformed_graph)

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'thresholds_selection')
        tb_w.add_all_statistics(transformed_graph, 'thresholds_selection')

    ######################################
    # Graph substitution (post statistics collection)
    ######################################
    transformed_graph = substitute(transformed_graph,
                                   fw_impl.get_substitutions_post_statistics_collection(
                                       core_config.quantization_config))

    ######################################
    # Shift Negative Activations
    ######################################
    if core_config.quantization_config.shift_negative_activation_correction:
        transformed_graph = fw_impl.shift_negative_correction(transformed_graph,
                                                              core_config,
                                                              fw_info)
        if tb_w is not None:
            tb_w.add_graph(transformed_graph, 'after_shift_negative_correction')
            tb_w.add_all_statistics(transformed_graph, 'after_shift_negative_correction')

    if tb_w is not None:
        tb_w.add_graph(transformed_graph, 'post_statistics_collection_substitutions')
        tb_w.add_all_statistics(transformed_graph, 'post_statistics_collection_substitutions')

    ######################################
    # Statistics Correction
    ######################################
    tg_with_bias = statistics_correction_runner(transformed_graph, core_config, fw_info, fw_impl, tb_w)

    for n in tg_with_bias.nodes:
        assert n.final_weights_quantization_cfg is None

    ######################################
    # Finalize bit widths
    ######################################
    if target_resource_utilization is not None:
        assert core_config.is_mixed_precision_enabled
        if core_config.mixed_precision_config.configuration_overwrite is None:

            bit_widths_config = search_bit_width(tg_with_bias,
                                                 fw_info,
                                                 fw_impl,
                                                 target_resource_utilization,
                                                 core_config.mixed_precision_config,
                                                 representative_data_gen)
        else:
            bit_widths_config = core_config.mixed_precision_config.configuration_overwrite

    else:
        bit_widths_config = []

    tg = set_bit_widths(core_config.is_mixed_precision_enabled,
                        tg_with_bias,
                        bit_widths_config)

    # Edit the graph again after finalizing the configurations.
    # This is since some actions regard the final configuration and should be edited.
    edit_network_graph(tg, fw_info, core_config.debug_config.network_editor)

    return tg


class BaseChangeQuantConfigAttrTest(BaseKerasFeatureNetworkTest):

    def __init__(self, unit_test, edit_filter, action, prepare_graph_func):
        self.edit_filter = edit_filter
        self.action = action
        self.prepare_graph_func = prepare_graph_func
        super().__init__(unit_test)

    def get_debug_config(self):
        return DebugConfig(network_editor=[EditRule(filter=self.edit_filter,
                                                    action=self.action)])

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(3, 4, use_bias=False)(inputs)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def run_test(self):
        feature_networks = self.create_networks()
        feature_networks = feature_networks if isinstance(feature_networks, list) else [feature_networks]
        for model_float in feature_networks:
            core_config = self.get_core_config()
            ptq_graph = self.prepare_graph_func(in_model=model_float,
                                                representative_data_gen=
                                                self.representative_data_gen_experimental,
                                                core_config=core_config,
                                                fw_info=self.get_fw_info(),
                                                fw_impl=self.get_fw_impl(),
                                                target_resource_utilization=self.get_resource_utilization(),
                                                tpc=self.get_tpc())

            filtered_nodes = ptq_graph.filter(self.edit_filter)
            for node in filtered_nodes:
                if node.final_weights_quantization_cfg is not None:
                    if hasattr(node.final_weights_quantization_cfg, list(self.action.kwargs.keys())[0]):
                        self.unit_test.assertTrue(node.final_weights_quantization_cfg.__getattribute__(
                            list(self.action.kwargs.keys())[0]) == list(self.action.kwargs.values())[0])
                elif node.final_activation_quantization_cfg is not None:
                    if hasattr(node.final_activation_quantization_cfg, list(self.action.kwargs.keys())[0]):
                        self.unit_test.assertTrue(node.final_activation_quantization_cfg.__getattribute__(
                            list(self.action.kwargs.keys())[0]) == list(self.action.kwargs.values())[0])
                else:
                    for nqc in node.candidates_quantization_cfg:
                        if hasattr(nqc.weights_quantization_cfg, list(self.action.kwargs.keys())[0]):
                            self.unit_test.assertTrue(nqc.weights_quantization_cfg.__getattribute__(
                                list(self.action.kwargs.keys())[0]) == list(self.action.kwargs.values())[0])
                        if hasattr(nqc.activation_quantization_cfg, list(self.action.kwargs.keys())[0]):
                            self.unit_test.assertTrue(nqc.activation_quantization_cfg.__getattribute__(
                                list(self.action.kwargs.keys())[0]) == list(self.action.kwargs.values())[0])


class ChangeCandidatesWeightsQuantConfigAttrTest(BaseChangeQuantConfigAttrTest):

    def __init__(self, unit_test):
        edit_filter = NodeTypeFilter(layers.Conv2D)
        action = ChangeCandidatesWeightsQuantConfigAttr(weights_bias_correction=False)
        prepare_graph_func = prepare_graph_for_first_network_editor
        super().__init__(unit_test, edit_filter=edit_filter, action=action, prepare_graph_func=prepare_graph_func)


class ChangeCandidatesActivationQCAttrTest(BaseChangeQuantConfigAttrTest):

    def __init__(self, unit_test):
        edit_filter = NodeTypeFilter(layers.Conv2D)
        action = ChangeCandidatesActivationQuantConfigAttr(activation_n_bits=7)
        prepare_graph_func = prepare_graph_for_first_network_editor
        super().__init__(unit_test, edit_filter=edit_filter, action=action, prepare_graph_func=prepare_graph_func)


class ChangeFinalsWeightsQuantConfigAttrTest(BaseChangeQuantConfigAttrTest):

    def __init__(self, unit_test):
        edit_filter = NodeTypeFilter(layers.Conv2D)
        action = ChangeFinalWeightsQuantConfigAttr(weights_bias_correction=False)
        prepare_graph_func = prepare_graph_for_second_network_editor
        super().__init__(unit_test, edit_filter=edit_filter, action=action, prepare_graph_func=prepare_graph_func)


class ChangeFinalsActivationQCAttrTest(BaseChangeQuantConfigAttrTest):

    def __init__(self, unit_test):
        edit_filter = NodeTypeFilter(layers.Conv2D)
        action = ChangeCandidatesActivationQuantConfigAttr(activation_n_bits=7)
        prepare_graph_func = prepare_graph_for_second_network_editor
        super().__init__(unit_test, edit_filter=edit_filter, action=action, prepare_graph_func=prepare_graph_func)


class BaseChangeQuantizationMethodQCAttrTest(BaseKerasFeatureNetworkTest):

    def __init__(self, unit_test, edit_filter, action, prepare_graph_func):
        self.edit_filter = edit_filter
        self.action = action
        self.prepare_graph_func = prepare_graph_func
        super().__init__(unit_test)

    def get_debug_config(self):
        return DebugConfig(network_editor=[EditRule(filter=self.edit_filter,
                                                    action=self.action)])

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(3, 4, use_bias=False)(inputs)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def run_test(self):
        feature_networks = self.create_networks()
        feature_networks = feature_networks if isinstance(feature_networks, list) else [feature_networks]
        for model_float in feature_networks:
            core_config = self.get_core_config()
            ptq_graph = self.prepare_graph_func(in_model=model_float,
                                                representative_data_gen=
                                                self.representative_data_gen_experimental,
                                                core_config=core_config,
                                                fw_info=self.get_fw_info(),
                                                fw_impl=self.get_fw_impl(),
                                                target_resource_utilization=self.get_resource_utilization(),
                                                tpc=self.get_tpc())

            filtered_nodes = ptq_graph.filter(self.edit_filter)
            for node in filtered_nodes:
                if node.final_weights_quantization_cfg is not None and hasattr(self.action,
                                                                               'weights_quantization_method'):
                    self.unit_test.assertTrue(node.final_weights_quantization_cfg.get_attr_config(KERNEL)
                                              .weights_quantization_method == self.action.weights_quantization_method)
                elif node.final_activation_quantization_cfg is not None and hasattr(self.action,
                                                                                    'activation_quantization_method'):
                    self.unit_test.assertTrue(node.final_activation_quantization_cfg.activation_quantization_method
                                              == self.action.activation_quantization_method)
                else:
                    for nqc in node.candidates_quantization_cfg:
                        if hasattr(self.action, 'activation_quantization_method'):
                            self.unit_test.assertTrue(nqc.activation_quantization_cfg.activation_quantization_method
                                                      == self.action.activation_quantization_method)
                        if hasattr(self.action, 'weights_quantization_method'):
                            self.unit_test.assertTrue(nqc.weights_quantization_cfg.get_attr_config(KERNEL)
                                                      .weights_quantization_method ==
                                                      self.action.weights_quantization_method)


class ChangeCandidatesActivationQuantizationMethodQCAttrTest(BaseChangeQuantizationMethodQCAttrTest):

    def __init__(self, unit_test):
        edit_filter = NodeTypeFilter(layers.Conv2D)
        action = ChangeCandidatesActivationQuantizationMethod(activation_quantization_method=QuantizationMethod.UNIFORM)
        prepare_graph_func = prepare_graph_for_first_network_editor
        super().__init__(unit_test, edit_filter=edit_filter, action=action, prepare_graph_func=prepare_graph_func)


class ChangeCandidatesWeightsQuantizationMethodQCAttrTest(BaseChangeQuantizationMethodQCAttrTest):

    def __init__(self, unit_test):
        edit_filter = NodeTypeFilter(layers.Conv2D)
        action = ChangeCandidatesWeightsQuantizationMethod(attr_name=KERNEL,
                                                           weights_quantization_method=QuantizationMethod.UNIFORM)
        prepare_graph_func = prepare_graph_for_first_network_editor
        super().__init__(unit_test, edit_filter=edit_filter, action=action, prepare_graph_func=prepare_graph_func)


class ChangeFinalsActivationQuantizationMethodQCAttrTest(BaseChangeQuantizationMethodQCAttrTest):

    def __init__(self, unit_test):
        edit_filter = NodeTypeFilter(layers.Conv2D)
        action = ChangeFinalActivationQuantizationMethod(activation_quantization_method=QuantizationMethod.UNIFORM)
        prepare_graph_func = prepare_graph_for_second_network_editor
        super().__init__(unit_test, edit_filter=edit_filter, action=action, prepare_graph_func=prepare_graph_func)


class ChangeFinalsWeightsQuantizationMethodQCAttrTest(BaseChangeQuantizationMethodQCAttrTest):

    def __init__(self, unit_test):
        edit_filter = NodeTypeFilter(layers.Conv2D)
        action = ChangeFinalWeightsQuantizationMethod(attr_name=KERNEL,
                                                      weights_quantization_method=QuantizationMethod.UNIFORM)
        prepare_graph_func = prepare_graph_for_second_network_editor
        super().__init__(unit_test, edit_filter=edit_filter, action=action, prepare_graph_func=prepare_graph_func)
