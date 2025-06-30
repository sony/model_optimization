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
import numpy as np

from tests_pytest._test_util.graph_builder_utils import build_node, build_nbits_qc
from model_compression_toolkit.core.common.framework_info import set_fw_info


def test_find_min_max_candidate_index(fw_info_mock):
    set_fw_info(fw_info_mock)
    qcs = []
    for ab in [4, 8, 16, 2]:
        for fb in [2, 8, 4]:
            for bb in [3, 5]:
                for pb in [15, 7, 11]:
                    qcs.append(build_nbits_qc(a_nbits=ab,
                                              w_attr={'foo': (fb, True), 'bar': (bb, True)},
                                              pos_attr=(pb, True, [2])))

    n = build_node('n', canonical_weights={'foo': np.random.random((5, 5)),
                                           'bar': np.random.random((6, 6)),
                                           2: np.random.random((6, 5))}, qcs=qcs)

    max_ind = n.find_max_candidate_index()
    max_qc = n.candidates_quantization_cfg[max_ind]
    assert max_qc.activation_quantization_cfg.activation_n_bits == 16
    assert max_qc.weights_quantization_cfg.get_attr_config('foo').weights_n_bits == 8
    assert max_qc.weights_quantization_cfg.get_attr_config('bar').weights_n_bits == 5
    assert max_qc.weights_quantization_cfg.pos_attributes_config_mapping[2].weights_n_bits == 15

    min_ind = n.find_min_candidate_index()
    min_qc = n.candidates_quantization_cfg[min_ind]
    assert min_qc.activation_quantization_cfg.activation_n_bits == 2
    assert min_qc.weights_quantization_cfg.get_attr_config('foo').weights_n_bits == 2
    assert min_qc.weights_quantization_cfg.get_attr_config('bar').weights_n_bits == 3
    assert min_qc.weights_quantization_cfg.pos_attributes_config_mapping[2].weights_n_bits == 7
