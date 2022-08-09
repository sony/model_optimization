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
import model_compression_toolkit as mct

tp = mct.target_platform


def generate_tp_model_with_activation_mp(base_cfg, mp_bitwidth_candidates_list, name="activation_mp_model"):

    # prepare mp candidates
    mixed_precision_cfg_list = []
    for weights_n_bits, activation_n_bits in mp_bitwidth_candidates_list:
        candidate_cfg = base_cfg.clone_and_edit(weights_n_bits=weights_n_bits,
                                                activation_n_bits=activation_n_bits)
        mixed_precision_cfg_list.append(candidate_cfg)

    # set hw model
    default_configuration_options = tp.QuantizationConfigOptions([base_cfg])

    generated_tp = tp.TargetPlatformModel(default_configuration_options, name=name)

    with generated_tp:
        tp.OperatorsSet("NoQuantization",
                         tp.get_default_quantization_config_options().clone_and_edit(
                             enable_weights_quantization=False,
                             enable_activation_quantization=False))

        mixed_precision_configuration_options = tp.QuantizationConfigOptions(mixed_precision_cfg_list,
                                                                              base_config=base_cfg)

        tp.OperatorsSet("Weights_n_Activation", mixed_precision_configuration_options)
        tp.OperatorsSet("Activation", mixed_precision_configuration_options)

    return generated_tp
