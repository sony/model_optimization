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
import pytest
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as current_schema
import model_compression_toolkit.target_platform_capabilities.schema.v2 as schema_v2
from model_compression_toolkit.core.common import BaseNode

from tests.common_tests.helpers.generate_test_tpc import generate_test_attr_configs, generate_test_op_qc

# Setup TEST_QC and TEST_QCO for testing.
TEST_QC = generate_test_op_qc(**generate_test_attr_configs())


class TestFusing:
    def test_fusing_single_opset(self):
        """Tests that creating a Fusing pattern with a single operator raises an exception."""
        add = current_schema.OperatorsSet(name="add")
        with pytest.raises(Exception, match="Fusing cannot be created for a single operator."):
            schema_v2.TargetPlatformCapabilities(
                default_qco=current_schema.QuantizationConfigOptions(quantization_configurations=(TEST_QC,)),
                operator_set=(add,),
                fusing_patterns=(schema_v2.Fusing(operator_groups=(add,)),),
                tpc_minor_version=None,
                tpc_patch_version=None,
                tpc_platform_type=None,
                add_metadata=False
            )

    def test_fusing_contains(self):
        """Tests that the contains method for fusing patterns correctly identifies containment relationships."""
        operator_set = []
        conv = current_schema.OperatorsSet(name="conv")
        add = current_schema.OperatorsSet(name="add")
        tanh = current_schema.OperatorsSet(name="tanh")
        operator_set.extend([conv, add, tanh])
        fusing_patterns = (
            schema_v2.Fusing(operator_groups=(conv, add)),
            schema_v2.Fusing(operator_groups=(conv, add, tanh))
        )
        hm = schema_v2.TargetPlatformCapabilities(
            default_qco=current_schema.QuantizationConfigOptions(quantization_configurations=(TEST_QC,)),
            operator_set=tuple(operator_set),
            fusing_patterns=fusing_patterns,
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            add_metadata=False
        )
        assert len(hm.fusing_patterns) == 2
        f0, f1 = hm.fusing_patterns
        assert f1.contains(f0)
        assert not f0.contains(f1)
        assert f0.contains(f0)
        assert f1.contains(f1)

    def test_fusing_contains_with_opset_concat(self):
        """Tests that fusing patterns correctly evaluate containment when operator sets are concatenated."""
        operator_set = []
        conv = current_schema.OperatorsSet(name="conv")
        add = current_schema.OperatorsSet(name="add")
        tanh = current_schema.OperatorsSet(name="tanh")
        operator_set.extend([conv, add, tanh])
        add_tanh = current_schema.OperatorSetGroup(operators_set=[add, tanh])
        fusing_patterns = (
            schema_v2.Fusing(operator_groups=(conv, add)),
            schema_v2.Fusing(operator_groups=(conv, add_tanh)),
            schema_v2.Fusing(operator_groups=(conv, add, tanh))
        )
        hm = schema_v2.TargetPlatformCapabilities(
            default_qco=current_schema.QuantizationConfigOptions(quantization_configurations=(TEST_QC,)),
            operator_set=tuple(operator_set),
            fusing_patterns=fusing_patterns,
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            add_metadata=False
        )
        assert len(hm.fusing_patterns) == 3
        f0, f1, f2 = hm.fusing_patterns
        assert f0.contains(f0)
        assert f1.contains(f1)
        assert f2.contains(f2)
        assert f2.contains(f0)
        assert f1.contains(f0)
        assert not f0.contains(f1)
        assert not f0.contains(f2)
        assert not f2.contains(f1)
        assert not f1.contains(f2)

    def test_fusing_set_qconfig(self):
        """Tests that the contains method for fusing patterns correctly identifies containment relationships."""

        operator_set = []
        conv = current_schema.OperatorsSet(name="conv")
        add = current_schema.OperatorsSet(name="add")
        tanh = current_schema.OperatorsSet(name="tanh")
        operator_set.extend([conv, add, tanh])

        fusing_patterns = (
            schema_v2.Fusing(operator_groups=(conv, add), fuse_op_quantization_config=TEST_QC),
            schema_v2.Fusing(operator_groups=(conv, tanh))    ### Do not set fuse_op_quantization_config.
        )

        ### check initialization of fused operator quantization config
        assert fusing_patterns[1].fuse_op_quantization_config == None
        assert isinstance(fusing_patterns[0].fuse_op_quantization_config, current_schema.OpQuantizationConfig) == True
        assert fusing_patterns[0].fuse_op_quantization_config == TEST_QC

        hm = schema_v2.TargetPlatformCapabilities(
            default_qco=current_schema.QuantizationConfigOptions(quantization_configurations=(TEST_QC,)),
            operator_set=tuple(operator_set),
            fusing_patterns=fusing_patterns,
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            add_metadata=False
        )

        ### Checking the impact of added fuse_op_quantization_config.
        assert len(hm.fusing_patterns) == 2
        f0, f1 = hm.fusing_patterns
        assert not f1.contains(f0)
        assert not f0.contains(f1)
        assert f0.contains(f0)
        assert f1.contains(f1)