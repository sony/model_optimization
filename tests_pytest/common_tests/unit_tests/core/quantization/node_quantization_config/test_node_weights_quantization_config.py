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
from typing import List
from unittest.mock import Mock

import pytest

from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core.common.quantization.node_quantization_config import \
    NodeWeightsQuantizationConfig
from model_compression_toolkit.target_platform_capabilities import Signedness, OpQuantizationConfig
from model_compression_toolkit.target_platform_capabilities.constants import POSITIONAL_ATTR
from model_compression_toolkit.target_platform_capabilities.schema.v1 import AttributeQuantizationConfig


class TestPositionalWeightsAttrQuantizationConfig:
    def _create_weights_attr_quantization_config(self, weights_n_bits: int) -> AttributeQuantizationConfig:
        """
        Helper method to create a weights attribute quantization configuration.

        Args:
            weights_n_bits (int): Number of bits to use for quantizing weights.

        Returns:
            AttributeQuantizationConfig: Holds the quantization configuration of a weight attribute of a layer.
        """
        weights_attr_config = AttributeQuantizationConfig(
            weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
            weights_n_bits=weights_n_bits,
            weights_per_channel_threshold=False,
            enable_weights_quantization=True,
            lut_values_bitwidth=None)
        return weights_attr_config

    def _create_node_weights_op_cfg(
            self,
            pos_weight_attr: List[str],
            pos_weight_attr_config: List[AttributeQuantizationConfig],
            def_weight_attr_config: AttributeQuantizationConfig) -> OpQuantizationConfig:
        """
        Helper method to create a Node Weights OpQuantizationConfig with a default weights
        attribute config and a specific weight attribute.

        Args:
            pos_weight_attr (List[str]): List of names for specific weight attributes.
            pos_weight_attr_config (List[AttributeQuantizationConfig]): Corresponding list of quantization configs
                                                                    for the specific attributes.
            def_weight_attr_config (AttributeQuantizationConfig): Default quantization config for the weights.

        Returns:
            OpQuantizationConfig: Class to configure the quantization parameters of an operator.
        """
        attr_weights_configs_mapping = dict(zip(pos_weight_attr, pos_weight_attr_config))

        op_cfg = OpQuantizationConfig(
            default_weight_attr_config=def_weight_attr_config,
            attr_weights_configs_mapping=attr_weights_configs_mapping,
            activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
            activation_n_bits=8,
            supported_input_activation_n_bits=8,
            enable_activation_quantization=True,
            quantization_preserving=True,
            fixed_scale=None,
            fixed_zero_point=None,
            simd_size=None,
            signedness=Signedness.AUTO
        )
        return op_cfg

    def test_node_weights_quantization_config_op_cfg_mapping(self):
        """
        Test case for verifying that the positional weight attribute is correctly mapped and
        configured in the NodeWeightsQuantizationConfig.
        """
        positional_weight_attr = 0
        weights_n_bits = 8
        pos_weights_n_bits = 16

        def_weight_attr_config = self._create_weights_attr_quantization_config(weights_n_bits)
        pos_weight_attr_config = self._create_weights_attr_quantization_config(pos_weights_n_bits)

        # Ensure the configs have different weights bit widths.
        assert def_weight_attr_config.weights_n_bits != pos_weight_attr_config.weights_n_bits

        op_cfg = self._create_node_weights_op_cfg(pos_weight_attr=[POSITIONAL_ATTR],
                                                  pos_weight_attr_config=[pos_weight_attr_config],
                                                  def_weight_attr_config=def_weight_attr_config)

        # Check that positional weights attribute config differs from default config.
        assert op_cfg.default_weight_attr_config.weights_n_bits != op_cfg.attr_weights_configs_mapping[
            POSITIONAL_ATTR].weights_n_bits

        weights_quant_cfg = NodeWeightsQuantizationConfig(op_cfg=op_cfg,
                                                          weights_channels_axis=Mock(),
                                                          node_attrs_list=[positional_weight_attr])

        # Check if the positional weight attribute was properly assigned in the positional attributes configuration
        # mapping.
        assert weights_quant_cfg.pos_attributes_config_mapping[
                   positional_weight_attr].weights_n_bits == pos_weight_attr_config.weights_n_bits

        # Test using the positional attribute as the key rather than POS_ATTR; this mismatch should cause
        # NodeWeightsQuantizationConfig to fall back to the default weights attribute configuration instead of
        # applying the specific one.
        op_cfg = self._create_node_weights_op_cfg(pos_weight_attr=[str(positional_weight_attr)],
                                                  pos_weight_attr_config=[pos_weight_attr_config],
                                                  def_weight_attr_config=def_weight_attr_config)

        # Check that positional weights attribute config differs from default config.
        assert op_cfg.default_weight_attr_config.weights_n_bits != op_cfg.attr_weights_configs_mapping[
            str(positional_weight_attr)].weights_n_bits

        weights_quant_cfg = NodeWeightsQuantizationConfig(op_cfg=op_cfg,
                                                          weights_channels_axis=Mock(),
                                                          node_attrs_list=[positional_weight_attr])

        # Check if the positional weight attribute was properly assigned in the positional attributes configuration
        # mapping.
        assert weights_quant_cfg.pos_attributes_config_mapping[
                   positional_weight_attr].weights_n_bits == def_weight_attr_config.weights_n_bits

        # Add a second positional attribute with a different config.
        second_positional_weight_attr = POSITIONAL_ATTR + '_1'
        second_pos_weights_n_bits = 32
        second_pos_weight_attr_config = self._create_weights_attr_quantization_config(second_pos_weights_n_bits)

        # Confirm all three configs have different bit widths.
        assert pos_weight_attr_config.weights_n_bits != second_pos_weight_attr_config.weights_n_bits

        # Create op config with two positional attribute keys and their respective configs.
        op_cfg = self._create_node_weights_op_cfg(pos_weight_attr=[POSITIONAL_ATTR, second_positional_weight_attr],
                                                  pos_weight_attr_config=[pos_weight_attr_config,
                                                                          second_pos_weight_attr_config],
                                                  def_weight_attr_config=def_weight_attr_config)

        # Check the configs are correctly set and distinct from each other and from the default.
        assert op_cfg.default_weight_attr_config.weights_n_bits != op_cfg.attr_weights_configs_mapping[
            str(POSITIONAL_ATTR)].weights_n_bits
        assert op_cfg.default_weight_attr_config.weights_n_bits != op_cfg.attr_weights_configs_mapping[
            str(second_positional_weight_attr)].weights_n_bits
        assert op_cfg.attr_weights_configs_mapping[
                   str(POSITIONAL_ATTR)].weights_n_bits != op_cfg.attr_weights_configs_mapping[
                   str(second_positional_weight_attr)].weights_n_bits

        # Expect ValueError: multiple matching keys found for positional weights attribute.
        with pytest.raises(ValueError, match='Found multiple attribute in FQC OpConfig that are contained in the '
                                             'attribute name \'0\'.Please fix the FQC attribute names mapping such '
                                             'that each operator\'s attribute would have a unique matching name.'):
            NodeWeightsQuantizationConfig(op_cfg=op_cfg, weights_channels_axis=Mock(),
                                          node_attrs_list=[positional_weight_attr])
