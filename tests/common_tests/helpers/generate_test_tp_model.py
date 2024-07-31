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

from model_compression_toolkit.constants import FLOAT_BITWIDTH, ACTIVATION_N_BITS_ATTRIBUTE, \
    SUPPORTED_INPUT_ACTIVATION_NBITS_ATTRIBUTE
from model_compression_toolkit.target_platform_capabilities.constants import OPS_SET_LIST, KERNEL_ATTR, BIAS_ATTR, \
    WEIGHTS_N_BITS
from model_compression_toolkit.target_platform_capabilities.target_platform import OpQuantizationConfig, QuantizationConfigOptions
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs, generate_tp_model
import model_compression_toolkit as mct

tp = mct.target_platform

DEFAULT_WEIGHT_ATTR_CONFIG = 'default_weight_attr_config'
KERNEL_BASE_CONFIG = 'kernel_base_config'
BIAS_CONFIG = 'bias_config'


def generate_test_tp_model(edit_params_dict, name=""):
    # Add "supported_input_activation_n_bits" to match "activation_n_bits" if not defined.
    if ACTIVATION_N_BITS_ATTRIBUTE in edit_params_dict and SUPPORTED_INPUT_ACTIVATION_NBITS_ATTRIBUTE not in edit_params_dict:
        edit_params_dict[SUPPORTED_INPUT_ACTIVATION_NBITS_ATTRIBUTE] = (edit_params_dict[ACTIVATION_N_BITS_ATTRIBUTE],)
    base_config, op_cfg_list, default_config = get_op_quantization_configs()

    # separate weights attribute parameters from the requested param to edit
    weights_params_names = [name for name in tp.AttributeQuantizationConfig.__init__.__code__.co_varnames if name != 'self']
    weights_params = {k: v for k, v in edit_params_dict.items() if k in weights_params_names}
    rest_params = {k: v for k, v in edit_params_dict.items() if k not in list(weights_params.keys())}

    # this util function enables to edit only the kernel quantization params in the TPC,
    # because it's the most general use for it that we have in our tests.
    # editing other attribute's config require specific solution per test.
    attr_weights_configs_mapping = base_config.attr_weights_configs_mapping
    attr_weights_configs_mapping[KERNEL_ATTR] = \
        attr_weights_configs_mapping[KERNEL_ATTR].clone_and_edit(**weights_params)
    updated_config = base_config.clone_and_edit(attr_weights_configs_mapping=attr_weights_configs_mapping,
                                                **rest_params)

    # For the default config, we only update the non-weights attributes argument, since the behaviour for the weights
    # quantization is supposed to remain the default defined behavior
    updated_default_config = base_config.clone_and_edit(**rest_params)

    # the target platform model's options config list must contain the given base config
    # this method only used for non-mixed-precision tests
    op_cfg_list = [updated_config]

    return generate_tp_model(default_config=updated_default_config,
                             base_config=updated_config,
                             mixed_precision_cfg_list=op_cfg_list,
                             name=name)


def generate_mixed_precision_test_tp_model(base_cfg, default_config, mp_bitwidth_candidates_list, name=""):
    mp_op_cfg_list = []
    for weights_n_bits, activation_n_bits in mp_bitwidth_candidates_list:
        candidate_cfg = base_cfg.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: weights_n_bits}},
                                                activation_n_bits=activation_n_bits)

        if candidate_cfg == base_cfg:
            # the base config must be a reference of an instance in the cfg_list, so we put it and not the clone in the list.
            mp_op_cfg_list.append(base_cfg)
        else:
            mp_op_cfg_list.append(candidate_cfg)

    return generate_tp_model(default_config=default_config,
                             base_config=base_cfg,
                             mixed_precision_cfg_list=mp_op_cfg_list,
                             name=name)


def generate_tp_model_with_activation_mp(base_cfg, default_config, mp_bitwidth_candidates_list, custom_opsets=[],
                                         name="activation_mp_model"):
    mp_op_cfg_list = []
    for weights_n_bits, activation_n_bits in mp_bitwidth_candidates_list:

        candidate_cfg = base_cfg.clone_and_edit(attr_weights_configs_mapping=
                                                {KERNEL_ATTR: base_cfg.attr_weights_configs_mapping[KERNEL_ATTR]
                                                .clone_and_edit(weights_n_bits=weights_n_bits),
                                                 **{k: v for k, v in base_cfg.attr_weights_configs_mapping.items() if
                                                    k != KERNEL_ATTR}},
                                                activation_n_bits=activation_n_bits)
        if candidate_cfg == base_cfg:
            # the base config must be a reference of an instance in the cfg_list, so we put it and not the clone in the list.
            mp_op_cfg_list.append(base_cfg)
        else:
            mp_op_cfg_list.append(candidate_cfg)

    base_tp_model = generate_tp_model(default_config=default_config,
                                      base_config=base_cfg,
                                      mixed_precision_cfg_list=mp_op_cfg_list,
                                      name=name)

    mixed_precision_configuration_options = tp.QuantizationConfigOptions(mp_op_cfg_list,
                                                                         base_config=base_cfg)

    operator_sets_dict = {op_set.name: mixed_precision_configuration_options for op_set in base_tp_model.operator_set
                          if op_set.name is not "NoQuantization"}
    operator_sets_dict["Input"] = mixed_precision_configuration_options
    for c_ops in custom_opsets:
        operator_sets_dict[c_ops] = mixed_precision_configuration_options

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
            if operator_sets_dict is not None and operator_sets_dict.get(op_set.name) is not None:
                qc_options = operator_sets_dict[op_set.name]
            else:
                qc_options = op_set.qc_options

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
                      op_sets_to_layer_drop: Dict[str, List[Any]] = None,
                      attr_mapping: Dict[str, Dict] = {}):

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
            am = attr_mapping.get(op_set_name)
            tp.OperationsSetToLayers(op_set_name, layers, attr_mapping=am)

    return tpc


def generate_test_attr_configs(default_cfg_nbits: int = 8,
                               default_cfg_quantizatiom_method: tp.QuantizationMethod = tp.QuantizationMethod.POWER_OF_TWO,
                               kernel_cfg_nbits: int = 8,
                               kernel_cfg_quantizatiom_method: tp.QuantizationMethod = tp.QuantizationMethod.POWER_OF_TWO,
                               enable_kernel_weights_quantization: bool = True,
                               kernel_lut_values_bitwidth: int = None):

    default_weight_attr_config = tp.AttributeQuantizationConfig(
        weights_quantization_method=default_cfg_quantizatiom_method,
        weights_n_bits=default_cfg_nbits,
        weights_per_channel_threshold=False,
        enable_weights_quantization=False,
        lut_values_bitwidth=None)

    kernel_base_config = tp.AttributeQuantizationConfig(
        weights_quantization_method=kernel_cfg_quantizatiom_method,
        weights_n_bits=kernel_cfg_nbits,
        weights_per_channel_threshold=True,
        enable_weights_quantization=enable_kernel_weights_quantization,
        lut_values_bitwidth=kernel_lut_values_bitwidth)

    bias_config = tp.AttributeQuantizationConfig(
        weights_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        weights_n_bits=FLOAT_BITWIDTH,
        weights_per_channel_threshold=False,
        enable_weights_quantization=False,
        lut_values_bitwidth=None)

    return {DEFAULT_WEIGHT_ATTR_CONFIG: default_weight_attr_config,
            KERNEL_BASE_CONFIG: kernel_base_config,
            BIAS_CONFIG: bias_config}


def generate_test_op_qc(default_weight_attr_config: tp.AttributeQuantizationConfig,
                        kernel_base_config: tp.AttributeQuantizationConfig,
                        bias_config: tp.AttributeQuantizationConfig,
                        enable_activation_quantization: bool = True,
                        activation_n_bits: int = 8,
                        activation_quantization_method: tp.QuantizationMethod = tp.QuantizationMethod.POWER_OF_TWO):

    return tp.OpQuantizationConfig(enable_activation_quantization=enable_activation_quantization,
                                   default_weight_attr_config=default_weight_attr_config,
                                   attr_weights_configs_mapping={KERNEL_ATTR: kernel_base_config,
                                                                 BIAS_ATTR: bias_config},
                                   activation_n_bits=activation_n_bits,
                                   supported_input_activation_n_bits=activation_n_bits,
                                   activation_quantization_method=activation_quantization_method,
                                   quantization_preserving=False,
                                   fixed_scale=None,
                                   fixed_zero_point=None,
                                   simd_size=32)
