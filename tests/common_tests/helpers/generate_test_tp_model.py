# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
from model_compression_toolkit.tpc_models.default_tp_model.get_default_tp_model import import_model
import model_compression_toolkit as mct


tp = mct.target_platform

def get_tp_model(version: str = None):
    m = import_model(version)
    return m.generate_tp_model


def get_op_quantization_configs(version: str = None):
    m = import_model(version)
    base_config, mixed_precision_cfg_list = m.get_op_quantization_configs()
    return base_config, mixed_precision_cfg_list


def generate_test_tp_model(edit_params_dict, name="", version=None):
    base_config, op_cfg_list = get_op_quantization_configs()
    updated_config = base_config.clone_and_edit(**edit_params_dict)

    # the target platform model's options config list must contain the given base config
    # this method only used for non-mixed-precision tests
    op_cfg_list = [updated_config]

    generate_tp_model = get_tp_model(version)
    return generate_tp_model(default_config=updated_config,
                                   base_config=updated_config,
                                   mixed_precision_cfg_list=op_cfg_list,
                                   name=name)


def generate_mixed_precision_test_tp_model(base_cfg, mp_bitwidth_candidates_list, name="", version=None):
    mp_op_cfg_list = []
    for weights_n_bits, activation_n_bits in mp_bitwidth_candidates_list:
        candidate_cfg = base_cfg.clone_and_edit(weights_n_bits=weights_n_bits,
                                                activation_n_bits=activation_n_bits)
        mp_op_cfg_list.append(candidate_cfg)

    generate_tp_model = get_tp_model(version)
    return generate_tp_model(default_config=base_cfg,
                             base_config=base_cfg,
                             mixed_precision_cfg_list=mp_op_cfg_list,
                             name=name)
