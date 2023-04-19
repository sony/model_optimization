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
import copy
from typing import Dict, List, Any

from model_compression_toolkit.core.common.constants import OPS_SET_LIST
from model_compression_toolkit.core.common.target_platform import OpQuantizationConfig, QuantizationConfigOptions
from model_compression_toolkit.target_platform_capabilities.tpc_models.default_tpc.latest import get_op_quantization_configs, generate_tp_model
import model_compression_toolkit as mct

tp = mct.target_platform


def generate_test_tp_model(edit_params_dict, name=""):
    base_config, op_cfg_list = get_op_quantization_configs()
    updated_config = base_config.clone_and_edit(**edit_params_dict)

    # the target platform model's options config list must contain the given base config
    # this method only used for non-mixed-precision tests
    op_cfg_list = [updated_config]

    return generate_tp_model(default_config=updated_config,
                             base_config=updated_config,
                             mixed_precision_cfg_list=op_cfg_list,
                             name=name)


def generate_mixed_precision_test_tp_model(base_cfg, mp_bitwidth_candidates_list, name=""):
    mp_op_cfg_list = []
    for weights_n_bits, activation_n_bits in mp_bitwidth_candidates_list:
        candidate_cfg = base_cfg.clone_and_edit(weights_n_bits=weights_n_bits,
                                                activation_n_bits=activation_n_bits)
        mp_op_cfg_list.append(candidate_cfg)

    return generate_tp_model(default_config=base_cfg,
                             base_config=base_cfg,
                             mixed_precision_cfg_list=mp_op_cfg_list,
                             name=name)


def generate_tp_model_with_activation_mp(base_cfg, mp_bitwidth_candidates_list, name="activation_mp_model"):
    mp_op_cfg_list = []
    for weights_n_bits, activation_n_bits in mp_bitwidth_candidates_list:
        candidate_cfg = base_cfg.clone_and_edit(weights_n_bits=weights_n_bits,
                                                activation_n_bits=activation_n_bits)
        mp_op_cfg_list.append(candidate_cfg)

    base_tp_model = generate_tp_model(default_config=base_cfg,
                                      base_config=base_cfg,
                                      mixed_precision_cfg_list=mp_op_cfg_list,
                                      name=name)

    mixed_precision_configuration_options = tp.QuantizationConfigOptions(mp_op_cfg_list,
                                                                         base_config=base_cfg)

    operator_sets_dict = {op_set.name: mixed_precision_configuration_options for op_set in base_tp_model.operator_set
                          if op_set.name is not "NoQuantization"}
    operator_sets_dict["Input"] = mixed_precision_configuration_options

    return generate_custom_test_tp_model(name=name,
                                         base_cfg=base_cfg,
                                         base_tp_model=base_tp_model,
                                         operator_sets_dict=operator_sets_dict)


def generate_custom_test_tp_model(name: str,
                                  base_cfg: OpQuantizationConfig,
                                  base_tp_model: tp.TargetPlatformModel,
                                  operator_sets_dict: Dict[str, QuantizationConfigOptions] = None):

    default_configuration_options = tp.QuantizationConfigOptions([base_cfg])

    custom_tp_model = tp.TargetPlatformModel(default_configuration_options, name=name)

    with custom_tp_model:
        for op_set in base_tp_model.operator_set:
            # Add existing OperatorSets from base TP model
            qc_options = op_set.qc_options if \
                (operator_sets_dict is None or op_set.name not in operator_sets_dict) and \
                (op_set.get_info().get(OPS_SET_LIST) is None) \
                else operator_sets_dict[op_set.name]

            tp.OperatorsSet(op_set.name, qc_options)

        existing_op_sets_names = [op_set.name for op_set in base_tp_model.operator_set]
        for op_set_name, op_set_qc_options in operator_sets_dict.items():
            # Add new OperatorSets from the given operator_sets_dict
            if op_set_name not in existing_op_sets_names:
                tp.OperatorsSet(op_set_name, op_set_qc_options)

        for fusion in base_tp_model.fusing_patterns:
            tp.Fusing(fusion.operator_groups_list)

    return custom_tp_model


def generate_test_tpc(name: str,
                      tp_model: tp.TargetPlatformModel,
                      base_tpc: tp.TargetPlatformCapabilities,
                      op_sets_to_layer_add: Dict[str, List[Any]] = None,
                      op_sets_to_layer_drop: Dict[str, List[Any]] = None):

    op_set_to_layers_list = base_tpc.op_sets_to_layers.op_sets_to_layers
    op_set_to_layers_dict = {op_set.name: op_set.layers for op_set in op_set_to_layers_list}

    merged_dict = copy.deepcopy(op_set_to_layers_dict)

    # Add new keys and update existing keys from op_sets_to_layer_add
    if op_sets_to_layer_add is not None:
        for op_set_name, layers in op_sets_to_layer_add.items():
            merged_dict[op_set_name] = merged_dict.get(op_set_name, []) + layers

    # Remove values from existing key
    if op_sets_to_layer_drop is not None:
        merged_dict = {op_set_name: [l for l in layers if l not in op_sets_to_layer_drop.get(op_set_name, [])]
                       for op_set_name, layers in merged_dict.items()}
        # Remove empty op sets
        merged_dict = {op_set_name: layers for op_set_name, layers in merged_dict.items() if len(layers) == 0}

    tpc = tp.TargetPlatformCapabilities(tp_model, name=name)

    with tpc:
        for op_set_name, layers in merged_dict.items():
            tp.OperationsSetToLayers(op_set_name, layers)

    return tpc
