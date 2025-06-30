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

from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.keras.constants import FUNCTION

import torch

from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OperatorSetNames, \
    QuantizationConfigOptions
from model_compression_toolkit.target_platform_capabilities.schema.schema_functions import \
    get_config_options_by_operators_set
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
    AttachTpcToPytorch
from tests.common_tests.helpers.generate_test_tpc import generate_custom_test_tpc
from tests.common_tests.helpers.tpcs_for_tests.v3.tpc import get_tpc

get_op_set = lambda x, x_list: [op_set for op_set in x_list if op_set.name == x][0]


# TODO irena: this tests node.filter_node_qco_by_graph which is not used anyway. What is actually used is
#  filter_node_qco_by_graph in set_node_quantization_config which doesn't have unittests.
@unittest.skip("TODO filter configs")
class TestTorchQuantConfigFiltering(unittest.TestCase):

    @staticmethod
    def get_tpc_default_16bit():
        tpc = get_tpc()
        base_cfg_16 = [c for c in get_config_options_by_operators_set(tpc,
                                                                      OperatorSetNames.MUL).quantization_configurations
                       if c.activation_n_bits == 16][0].clone_and_edit()
        qco_16 = QuantizationConfigOptions(base_config=base_cfg_16,
                                           quantization_configurations=(tpc.default_qco.base_config,
                                                                        base_cfg_16))
        tpc = generate_custom_test_tpc(
            name="custom_16_bit_tpc",
            base_cfg=tpc.default_qco.base_config,
            base_tpc=tpc,
            operator_sets_dict={
                OperatorSetNames.MUL: qco_16,
                OperatorSetNames.GELU: qco_16,
                OperatorSetNames.TANH: qco_16,
            })

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
        tpc = AttachTpcToPytorch().attach(tpc)

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
