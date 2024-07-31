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
import copy
import unittest
from mct_quantizers import QuantizationMethod

from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import \
    NodeActivationQuantizationConfig, NodeWeightsQuantizationConfig, WeightsAttrQuantizationConfig
from model_compression_toolkit.core.common.quantization.quantization_params_generation import \
    power_of_two_selection_histogram
from model_compression_toolkit.core.common.quantization.quantizers.uniform_quantizers import power_of_two_quantizer
from model_compression_toolkit.core.keras.constants import KERNEL, BIAS
from model_compression_toolkit.target_platform_capabilities.target_platform import AttributeQuantizationConfig
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs


class TestNodeQuantizationConfigurations(unittest.TestCase):

    def test_activation_set_quant_config_attribute(self):
        qc = QuantizationConfig()
        op_cfg, _, _ = get_op_quantization_configs()

        nac = NodeActivationQuantizationConfig(qc, op_cfg,
                                               activation_quantization_fn=power_of_two_quantizer,
                                               activation_quantization_params_fn=power_of_two_selection_histogram)
        og_nac = copy.deepcopy(nac)

        self.assertTrue(nac.activation_n_bits == 8)
        nac.set_quant_config_attr("activation_n_bits", 4)
        self.assertTrue(nac.activation_n_bits == 4, "Expects set_quant_config_attr to be successful, "
                                                    "new activation_n_bits should be 4.")
        self.assertFalse(nac == og_nac)

        update_nac = copy.deepcopy(nac)

        nac.set_quant_config_attr("activation_M_bits", 8)
        self.assertFalse(nac.activation_n_bits == 8, "Expects set_quant_config_attr to not update, "
                                                     "activation_n_bits should be 4.")
        self.assertTrue(nac == update_nac)

    def test_weights_set_quant_config_attribute(self):
        qc = QuantizationConfig()
        op_cfg, _, _ = get_op_quantization_configs()

        nwc = NodeWeightsQuantizationConfig(qc, op_cfg,
                                            weights_channels_axis=(1, -1),
                                            node_attrs_list=[KERNEL, 0])
        og_nwc = copy.deepcopy(nwc)

        # Updating a config parameter, not weights attribute parameter (no attr_name passed)
        self.assertTrue(nwc.weights_bias_correction)
        nwc.set_quant_config_attr("weights_bias_correction", False)
        self.assertFalse(nwc.weights_bias_correction)
        self.assertFalse(nwc == og_nwc)

        nwc = copy.deepcopy(og_nwc)

        # Updating an attribute parameter
        self.assertTrue(nwc.get_attr_config(KERNEL).weights_n_bits, 8)
        nwc.set_quant_config_attr("weights_n_bits", 4, attr_name=KERNEL)
        self.assertFalse(nwc.get_attr_config(KERNEL).weights_n_bits == 8,
                         f"Expects set_quant_config_attr to update {KERNEL} attribute weights_n_bits to 4.")
        self.assertFalse(nwc == og_nwc)

        nwc = copy.deepcopy(og_nwc)
        self.assertTrue(nwc.get_attr_config(0).weights_n_bits, 8)
        nwc.set_quant_config_attr("weights_n_bits", 4, attr_name=0)
        self.assertFalse(nwc.get_attr_config(0).weights_n_bits == 8,
                         f"Expects set_quant_config_attr to update positional attribute weights_n_bits to 4.")
        self.assertFalse(nwc == og_nwc)

        # Updating an non-existing attribute parameter, no update expected
        nwc = copy.deepcopy(og_nwc)
        nwc.set_quant_config_attr("weights_M_bits", 4, attr_name=KERNEL)
        self.assertTrue(nwc.get_attr_config(KERNEL).weights_n_bits == 8,
                         f"Expects set_quant_config_attr to not update {KERNEL} attribute weights_n_bits to 4.")
        self.assertTrue(nwc == og_nwc)

    def test_get_weights_attr_config(self):
        qc = QuantizationConfig()
        op_cfg, _, _ = get_op_quantization_configs()

        # Init a config with regular and positional attributes, and attributes with overlapping names, since in the
        # implementation we look for existence of a string to retrieve attribute
        nwc = NodeWeightsQuantizationConfig(qc, op_cfg,
                                            weights_channels_axis=(1, -1),
                                            node_attrs_list=[KERNEL, 0, BIAS, f"{BIAS}-2"])

        kernel_attr = nwc.get_attr_config(KERNEL)
        self.assertTrue(kernel_attr.weights_n_bits == 8)  # sanity

        pos_attr = nwc.get_attr_config(0)
        self.assertTrue(pos_attr.weights_quantization_method == QuantizationMethod.POWER_OF_TWO)  # sanity (should use default config)

        bias_attr = nwc.get_attr_config(BIAS)
        self.assertTrue(bias_attr.weights_n_bits == 8)  # checking successful retrival

        bias2_attr = nwc.get_attr_config(f"{BIAS}-2")
        self.assertTrue(bias_attr.weights_n_bits == 8)  # checking successful retrival

        self.assertFalse(bias_attr is bias2_attr)  # this is "is" on purpose, to compare addresses

    def test_set_weights_attr_config(self):
        qc = QuantizationConfig()
        op_cfg, _, _ = get_op_quantization_configs()

        nwc = NodeWeightsQuantizationConfig(qc, op_cfg,
                                            weights_channels_axis=(1, -1),
                                            node_attrs_list=[KERNEL, 0])

        new_cfg = WeightsAttrQuantizationConfig(qc,
                                                weights_attr_cfg=AttributeQuantizationConfig(weights_n_bits=4))

        kernel_attr = copy.deepcopy(nwc.get_attr_config(KERNEL))
        nwc.set_attr_config(KERNEL, new_cfg)
        self.assertTrue(kernel_attr.weights_n_bits == 8)
        self.assertTrue(nwc.get_attr_config(KERNEL).weights_n_bits == 4)

        pos_attr = copy.deepcopy(nwc.get_attr_config(0))
        nwc.set_attr_config(0, new_cfg)
        self.assertTrue(pos_attr.weights_n_bits == 8)
        self.assertTrue(nwc.get_attr_config(0).weights_n_bits == 4)


if __name__ == '__main__':
    unittest.main()
