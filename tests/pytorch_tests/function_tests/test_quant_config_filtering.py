# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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


import unittest
import model_compression_toolkit as mct
from model_compression_toolkit.constants import PYTORCH
from model_compression_toolkit.target_platform_capabilities.constants import IMX500_TP_MODEL
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.keras.constants import FUNCTION

import torch


get_op_set = lambda x, x_list: [op_set for op_set in x_list if op_set.name == x][0]


class TestTorchQuantConfigFiltering(unittest.TestCase):

    @staticmethod
    def get_tpc_default_16bit():
        tpc = mct.get_target_platform_capabilities(PYTORCH, IMX500_TP_MODEL, 'v3')
        # Force Mul base_config to 16bit only
        mul_op_set = get_op_set('Mul', tpc.tp_model.operator_set)
        mul_op_set.qc_options.base_config = [l for l in mul_op_set.qc_options.quantization_config_list if l.activation_n_bits == 16][0]
        tpc.layer2qco[torch.multiply].base_config = mul_op_set.qc_options.base_config
        return tpc

    def test_config_filtering(self):
        node = FunctionalNode('node',{},
                              [1, 8], [1, 8], {}, torch.multiply,
                              [], {}, functional_op=torch.multiply)
        next_node1 = FunctionalNode('next_node',{FUNCTION},
                              [1, 8], [1, 8], {}, torch.add,
                              [], {}, functional_op=torch.add)
        next_node2 = FunctionalNode('next_node',{},
                              [1, 8], [1, 8], {}, torch.div,
                              [], {}, functional_op=torch.div)

        tpc = self.get_tpc_default_16bit()
        node_qc_options = node.get_qco(tpc)
        self.assertTrue(node_qc_options.base_config.activation_n_bits == 16, "base_config should start with 16 bits.")

        # test base_config changed due to next node supported input bits.
        base_config, node_qc_options_list = node.filter_node_qco_by_graph(tpc, [next_node2], node_qc_options)
        self.assertTrue(base_config.activation_n_bits == 8, "base_config should start with 16 bits.")
        self.assertTrue(len(node_qc_options_list) == 1, "One of the QC options should have been filtered.")

        # test base_config changed due to one of the next nodes supported input bits.
        base_config, node_qc_options_list = node.filter_node_qco_by_graph(tpc, [next_node1, next_node2], node_qc_options)
        self.assertTrue(base_config.activation_n_bits == 8, "base_config should start with 16 bits.")
        self.assertTrue(len(node_qc_options_list) == 1, "One of the QC options should have been filtered.")

        # test base_config not changed because next node supports input bits.
        base_config, node_qc_options_list = node.filter_node_qco_by_graph(tpc, [next_node1], node_qc_options)
        self.assertTrue(base_config.activation_n_bits == 16, "base_config should start with 16 bits.")
        self.assertTrue(len(node_qc_options_list) == 2, "No QC options should have been filtered.")


if __name__ == '__main__':
    unittest.main()
