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
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs, generate_tp_model
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
