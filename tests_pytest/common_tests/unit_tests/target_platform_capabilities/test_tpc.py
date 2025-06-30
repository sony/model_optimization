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
import os
import pytest
from pydantic import ValidationError
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as current_schema
import model_compression_toolkit.target_platform_capabilities.schema.v1 as schema_v1
import model_compression_toolkit.target_platform_capabilities.schema.v2 as schema_v2
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR
from model_compression_toolkit.target_platform_capabilities.schema.schema_compatability import ALL_SCHEMA_VERSIONS, \
    all_tpc_types, FUTURE_SCHEMA_VERSIONS, _schema_v1_to_v2
from model_compression_toolkit.target_platform_capabilities.schema.schema_functions import get_config_options_by_operators_set, is_opset_in_model
from model_compression_toolkit.target_platform_capabilities.tpc_io_handler import load_target_platform_capabilities, \
    export_target_platform_capabilities
from tests.common_tests.helpers.generate_test_tpc import generate_test_attr_configs, generate_test_op_qc

# Setup TEST_QC and TEST_QCO for testing.
TEST_QC = generate_test_op_qc(**generate_test_attr_configs())
TEST_QCO = current_schema.QuantizationConfigOptions(quantization_configurations=(TEST_QC,))


# Fixtures for reusable resources

'''
The tmp_path fixture is a built-in pytest fixture that provides a temporary directory unique 
to each test function. It is represented as a pathlib.Path object and is automatically created 
and cleaned up by pytest. You can use it to create temporary files and directories during tests 
without worrying about manual cleanup. For more details, please see the pytest tmp_path documentation.
'''


def get_tpc(selected_schema):
    """
    :param selected_schema: A schema to create tpc from
    :return: TargetPlatformCapabilities instance using the given selected_schema
    """
    op1 = selected_schema.OperatorsSet(name="opset1")
    op2 = selected_schema.OperatorsSet(name="opset2")
    op3 = selected_schema.OperatorsSet(name="opset3")
    op12 = selected_schema.OperatorSetGroup(operators_set=[op1, op2])
    return selected_schema.TargetPlatformCapabilities(
        default_qco=TEST_QCO,
        operator_set=(op1, op2, op3),
        fusing_patterns=(
            selected_schema.Fusing(operator_groups=(op12, op3)),
            selected_schema.Fusing(operator_groups=(op1, op2))
        ),
        tpc_minor_version=1,
        tpc_patch_version=0,
        tpc_platform_type="dump_to_json",
        add_metadata=False
    )


@pytest.fixture
def tpc():
    """Fixture that returns a TargetPlatformCapabilities instance of current schema."""
    return get_tpc(current_schema)


@pytest.fixture(params=ALL_SCHEMA_VERSIONS)
def tpc_by_schema_version(request):
    """Fixture that yields a TargetPlatformCapabilities instance for each schema version."""
    selected_schema = request.param
    yield get_tpc(selected_schema)


@pytest.fixture(params=ALL_SCHEMA_VERSIONS + FUTURE_SCHEMA_VERSIONS)
def tpc_by_future_schema_version(request):
    """Fixture that yields a TargetPlatformCapabilities instance for each schema version, including future schemas."""
    selected_schema = request.param
    yield get_tpc(selected_schema)


@pytest.fixture
def tmp_invalid_json(tmp_path):
    """Fixture that creates an invalid JSON file."""
    invalid_json = '{"field1": "value1", "field2": '  # Incomplete JSON
    file = tmp_path / "invalid_model.json"
    file.write_text(invalid_json)
    return file


@pytest.fixture
def valid_export_path(tmp_path):
    """Fixture that provides a valid export file path."""
    return tmp_path / "exported_model.json"


@pytest.fixture
def invalid_export_path(tmp_path):
    """Fixture that simulates an invalid export file path."""
    return "/invalid/path/exported_model.json"


@pytest.fixture
def nonexistent_file(tmp_path):
    """Fixture that returns a nonexistent file path."""
    return tmp_path / "nonexistent.json"


class TestTPModelInitialization:

    def test_tpc_initialization(self, tpc_by_future_schema_version):
        """
        Test tpc initialization for all supported schemas & future schemas (i.e. schema with newer version
        than current schema). Test goal is to validate new (not in use yet) schemas are covered by Coverage test
        """
        tpc_by_schema = tpc_by_future_schema_version


class TestSchemaCompatability:

    def test_schema_compatibility(self, tpc_by_schema_version):
        """Tests that tpc of any schema version is supported and can be converted into current schema version"""
        tpc_by_schema = tpc_by_schema_version
        result = load_target_platform_capabilities(tpc_by_schema)
        assert isinstance(result, current_schema.TargetPlatformCapabilities), \
            f"Make sure all schema versions can be converted into current schema version"

    def test_schema_v1_to_schema_v2_compatability(self):
        """
        Test that tpc of schema v1 doesn't have the new fields added in schema v2, and that after converting it to
        schema v2 the new fields exist with the correct default value.
        """
        tpc_schema_v1 = get_tpc(schema_v1)
        with pytest.raises(AttributeError, match="'TargetPlatformCapabilities' object has no attribute 'insert_preserving_quantizers'"):
            tpc_schema_v1.insert_preserving_quantizers
        with pytest.raises(AttributeError, match="'Fusing' object has no attribute 'fuse_op_quantization_config'"):
            tpc_schema_v1.fusing_patterns[0].fuse_op_quantization_config
        tpc_schema_v2 = _schema_v1_to_v2(tpc_schema_v1)
        assert isinstance(tpc_schema_v2, schema_v2.TargetPlatformCapabilities)
        assert tpc_schema_v2.insert_preserving_quantizers is False
        assert tpc_schema_v2.fusing_patterns[0].fuse_op_quantization_config.enable_activation_quantization is False

    def test_json_schema_compatibility(self, tpc, valid_export_path):
        """Tests that a tpc exported using schema v1 is imported as current schema tpc."""
        tpc_schema_v1 = get_tpc(schema_v1)
        assert isinstance(tpc_schema_v1, schema_v1.TargetPlatformCapabilities)
        export_target_platform_capabilities(tpc_schema_v1, str(valid_export_path))
        imported_model = load_target_platform_capabilities(str(valid_export_path))
        assert isinstance(imported_model, current_schema.TargetPlatformCapabilities)


class TestTPModelInputOutput:
    def test_valid_model_object(self, tpc):
        # Tests that a valid TPC object is returned unchanged.
        result = load_target_platform_capabilities(tpc)
        assert result == tpc

    def test_new_schema(self, tpc):
        """Tests that current schema is in all schemas list. This test validates new schema was added properly."""
        assert type(tpc) in all_tpc_types, "Current schema need to be added to ALL_SCHEMA_VERSIONS"

    def test_invalid_json_parsing(self, tmp_invalid_json):
        """Tests that invalid JSON content raises a ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON for loading TargetPlatformCapabilities in"):
            load_target_platform_capabilities(str(tmp_invalid_json))

    def test_nonexistent_file(self, nonexistent_file):
        """Tests that a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="is not a valid file"):
            load_target_platform_capabilities(str(nonexistent_file))

    def test_non_json_extension(self, tmp_path, tmp_invalid_json):
        """Tests that a file with a non-JSON extension raises ValueError."""
        non_json_file = tmp_path / "test_model.txt"
        non_json_file.write_text(tmp_invalid_json.read_text())
        with pytest.raises(ValueError, match="does not have a '.json' extension"):
            load_target_platform_capabilities(str(non_json_file))
        non_json_file.unlink()

    def test_invalid_input_type(self):
        """Tests that an unsupported input type raises TypeError."""
        with pytest.raises(TypeError, match="must be either a TargetPlatformCapabilities instance or a string path"):
            load_target_platform_capabilities(123)

    def test_valid_export(self, tpc, valid_export_path):
        """Tests exporting a valid TargetPlatformCapabilities instance to a file."""
        export_target_platform_capabilities(tpc, str(valid_export_path))
        assert os.path.exists(str(valid_export_path))
        with open(str(valid_export_path), "r", encoding="utf-8") as file:
            content = file.read()
        assert content == tpc.model_dump_json(indent=4)

    def test_export_with_invalid_model(self, valid_export_path):
        """Tests that exporting an invalid model raises a ValueError."""
        with pytest.raises(ValueError, match="not a valid TargetPlatformCapabilities instance"):
            export_target_platform_capabilities("not_a_model", str(valid_export_path))

    def test_export_with_invalid_path(self, tpc, invalid_export_path):
        """Tests that exporting to an invalid path raises an OSError."""
        with pytest.raises(OSError, match="Failed to write to file"):
            export_target_platform_capabilities(tpc, str(invalid_export_path))

    def test_export_creates_parent_directories(self, tpc, tmp_path):
        """Tests that exporting creates parent directories as needed."""
        nested_path = tmp_path / "nested" / "directory" / "exported_model.json"
        export_target_platform_capabilities(tpc, str(nested_path))
        assert os.path.exists(str(nested_path))
        with open(str(nested_path), "r", encoding="utf-8") as file:
            content = file.read()
        assert content == tpc.model_dump_json(indent=4)
        # Cleanup created directories
        os.remove(str(nested_path))
        os.rmdir(str(tmp_path / "nested" / "directory"))
        os.rmdir(str(tmp_path / "nested"))

    def test_export_then_import(self, tpc, valid_export_path):
        """Tests that a model exported and then imported is identical."""
        export_target_platform_capabilities(tpc, str(valid_export_path))
        imported_model = load_target_platform_capabilities(str(valid_export_path))
        assert imported_model == tpc


class TestTargetPlatformModeling:
    def test_immutable_tp(self):
        """Tests that modifying an immutable TargetPlatformCapabilities instance raises an exception."""
        model = current_schema.TargetPlatformCapabilities(
            default_qco=TEST_QCO,
                operator_set=(current_schema.OperatorsSet(name="opset"),),
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            add_metadata=False
        )
        # Expecting a TypeError or AttributeError due to immutability
        with pytest.raises(ValidationError , match="Instance is frozen"):
            model.operator_set = tuple()

    def test_default_options_more_than_single_qc(self):
        """Tests that creating a TargetPlatformCapabilities with default_qco containing more than one configuration raises an exception."""
        test_qco = current_schema.QuantizationConfigOptions(
            quantization_configurations=(TEST_QC, TEST_QC),
            base_config=TEST_QC
        )
        with pytest.raises(Exception, match='Default QuantizationConfigOptions must contain exactly one option.'):
            current_schema.TargetPlatformCapabilities(
                default_qco=test_qco,
                tpc_minor_version=None,
                tpc_patch_version=None,
                tpc_platform_type=None,
                add_metadata=False
            )

    def test_tpc_show(self, capsys):
        """Tests that the show() method of TargetPlatformCapabilities produces output."""
        tpm = current_schema.TargetPlatformCapabilities(
            default_qco=TEST_QCO,
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            operator_set=(current_schema.OperatorsSet(name="opA"), current_schema.OperatorsSet(name="opB")),
            fusing_patterns=(current_schema.Fusing(operator_groups=(current_schema.OperatorsSet(name="opA"), current_schema.OperatorsSet(name="opB"))),),
            add_metadata=False
        )
        tpm.show()
        captured = capsys.readouterr().out
        assert captured  # Verify that output was printed


class TestOpset:
    def test_opset_qco(self):
        """Tests that the quantization configuration options for an opset are correctly set and retrievable."""
        opset_name = "ops_3bit"
        qco_3bit = TEST_QCO.clone_and_edit(activation_n_bits=3)
        operator_set = [current_schema.OperatorsSet(name=opset_name, qc_options=qco_3bit)]
        hm = current_schema.TargetPlatformCapabilities(
            default_qco=TEST_QCO,
            operator_set=tuple(operator_set),
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            add_metadata=False,
            name='test'
        )
        for op_qc in get_config_options_by_operators_set(hm, opset_name).quantization_configurations:
            assert op_qc.activation_n_bits == 3
        assert is_opset_in_model(hm, opset_name)
        assert not is_opset_in_model(hm, "ShouldNotBeInModel")
        assert get_config_options_by_operators_set(hm, opset_name) == qco_3bit
        assert get_config_options_by_operators_set(hm, "ShouldNotBeInModel") == hm.default_qco

    def test_opset_concat(self):
        """Tests that concatenation of operator sets is handled correctly and non-existent concatenated opsets are not found."""
        operator_set = []
        a = current_schema.OperatorsSet(name='opset_A')
        b = current_schema.OperatorsSet(name='opset_B', qc_options=TEST_QCO.clone_and_edit(activation_n_bits=2))
        c = current_schema.OperatorsSet(name='opset_C')
        operator_set.extend([a, b, c])
        hm = current_schema.TargetPlatformCapabilities(
            default_qco=TEST_QCO,
            operator_set=tuple(operator_set),
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            add_metadata=False,
            name='test'
        )
        assert len(hm.operator_set) == 3
        assert not is_opset_in_model(hm, "opset_A_opset_B")

    def test_non_unique_opset(self):
        """Tests that creating a TargetPlatformCapabilities with duplicate operator set names raises an exception."""
        with pytest.raises(Exception, match='Operator Sets must have unique names.'):
            current_schema.TargetPlatformCapabilities(
                default_qco=current_schema.QuantizationConfigOptions(quantization_configurations=(TEST_QC,)),
                operator_set=(current_schema.OperatorsSet(name="conv"), current_schema.OperatorsSet(name="conv")),
                tpc_minor_version=None,
                tpc_patch_version=None,
                tpc_platform_type=None,
                add_metadata=False
            )


class TestQCOptions:
    def test_empty_qc_options(self):
        """Tests that initializing QuantizationConfigOptions with an empty configuration raises an exception."""
        with pytest.raises(Exception, match="'QuantizationConfigOptions' requires at least one 'OpQuantizationConfig'. The provided configurations are empty."):
            current_schema.QuantizationConfigOptions(quantization_configurations=())

    def test_list_of_no_qc(self):
        """Tests that providing an invalid configuration list (non-dict values) to QuantizationConfigOptions raises an exception."""
        with pytest.raises(ValidationError, match="Input should be a valid dictionary"):
            current_schema.QuantizationConfigOptions(quantization_configurations=(TEST_QC, 3), base_config=TEST_QC)

    def test_clone_and_edit_options(self):
        """Tests that the clone_and_edit methods correctly modify the quantization configuration options."""
        modified_options = TEST_QCO.clone_and_edit(activation_n_bits=3).clone_and_edit_weight_attribute(
            attrs=[KERNEL_ATTR],
            weights_n_bits=5
        )
        assert modified_options.quantization_configurations[0].activation_n_bits == 3
        assert modified_options.quantization_configurations[0].attr_weights_configs_mapping[KERNEL_ATTR].weights_n_bits == 5

    def test_qco_without_base_config(self):
        """Tests that providing multiple configurations without a base_config raises an exception."""
        # Single config should work
        current_schema.QuantizationConfigOptions(quantization_configurations=(TEST_QC,))
        with pytest.raises(Exception, match="For multiple configurations, a 'base_config' is required for non-mixed-precision optimization."):
            current_schema.QuantizationConfigOptions(quantization_configurations=(TEST_QC, TEST_QC))


class TestFusing:
    def test_fusing_single_opset(self):
        """Tests that creating a Fusing pattern with a single operator raises an exception."""
        add = current_schema.OperatorsSet(name="add")
        with pytest.raises(Exception, match="Fusing cannot be created for a single operator."):
            current_schema.TargetPlatformCapabilities(
                default_qco=current_schema.QuantizationConfigOptions(quantization_configurations=(TEST_QC,)),
                operator_set=(add,),
                fusing_patterns=(current_schema.Fusing(operator_groups=(add,)),),
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
            current_schema.Fusing(operator_groups=(conv, add)),
            current_schema.Fusing(operator_groups=(conv, add, tanh))
        )
        hm = current_schema.TargetPlatformCapabilities(
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
            current_schema.Fusing(operator_groups=(conv, add)),
            current_schema.Fusing(operator_groups=(conv, add_tanh)),
            current_schema.Fusing(operator_groups=(conv, add, tanh))
        )
        hm = current_schema.TargetPlatformCapabilities(
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

