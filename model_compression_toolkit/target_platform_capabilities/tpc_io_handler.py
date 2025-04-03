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
from pathlib import Path
from typing import Union, Any

from model_compression_toolkit.target_platform_capabilities.schema.v1 import TargetPlatformCapabilities as schema_v1
from model_compression_toolkit.target_platform_capabilities.schema.v2 import TargetPlatformCapabilities as schema_v2
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema


SCHEMA_VERSIONS = [schema_v1, schema_v2]


def _is_tpc_instance(tpc_obj_or_path: Any):
    return type(tpc_obj_or_path) in [schema_v1, schema_v2]


def _tpc_to_current_schema_version(tpc: schema.TargetPlatformCapabilities):
    while tpc.SCHEMA_VERSION != schema.TargetPlatformCapabilities.SCHEMA_VERSION:
        tpc = tpc.to_next_version()
    return tpc


def _is_valid_tpc(tpc: schema.TargetPlatformCapabilities):
    for op in tpc.operator_set:
        if op.name not in schema.OperatorSetNames.get_values():
            raise TypeError(f"TPC contains unsupported operator {op.name}")
    return True


def _get_tpc_from_json(tpc_path):
    path = Path(tpc_path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"The path '{tpc_path}' is not a valid file.")
    # Verify that the file has a .json extension
    if path.suffix.lower() != '.json':
        raise ValueError(f"The file '{path}' does not have a '.json' extension.")
    try:
        with path.open('r', encoding='utf-8') as file:
            data = file.read()
    except OSError as e:
        raise ValueError(f"Error reading the file '{tpc_path}': {e.strerror}.") from e

    try:
        return schema.TargetPlatformCapabilities.parse_raw(data)
    except ValueError as e:
        raise ValueError(f"Invalid JSON for loading TargetPlatformCapabilities in '{tpc_path}': {e}.") from e
    except Exception as e:
        raise ValueError(f"Unexpected error while initializing TargetPlatformCapabilities: {e}.") from e


def load_target_platform_capabilities(tpc_obj_or_path: Union[schema.TargetPlatformCapabilities, str]) -> schema.TargetPlatformCapabilities:
    """
        Parses the tpc input, which can be either a TargetPlatformCapabilities object
        or a string path to a JSON file.

        Parameters:
            tpc_obj_or_path (Union[TargetPlatformCapabilities, str]): Input target platform model or path to .JSON file.

        Returns:
            TargetPlatformCapabilities: The parsed TargetPlatformCapabilities.

        Raises:
            FileNotFoundError: If the JSON file does not exist.
            ValueError: If the JSON content is invalid or cannot initialize the TargetPlatformCapabilities.
            TypeError: If the input is neither a TargetPlatformCapabilities nor a valid JSON file path.
        """
    if _is_tpc_instance(tpc_obj_or_path):
        tpc = tpc_obj_or_path
    elif isinstance(tpc_obj_or_path, str):
        tpc = _get_tpc_from_json(tpc_obj_or_path)
    else:
        raise TypeError(
            f"tpc_obj_or_path must be either a TargetPlatformCapabilities instance or a string path to a JSON file, "
            f"but received type '{type(tpc_obj_or_path).__name__}'."
        )

    # if tpc.SCHEMA_VERSION == schema.TargetPlatformCapabilities.SCHEMA_VERSION:
    #     return tpc_obj_or_path
    if _is_valid_tpc(tpc):
        return _tpc_to_current_schema_version(tpc_obj_or_path)

    """
    questions for tommorow:
    1. if we apply parser simple tests will fail (e.g defining opset1, opset2, opset3)
    2. TargetPlatformCapabilities is of different type for each scehma - will need to add every new schema version to list
    3. what additional items we want to add to parser?
    4. creating new schema from old schema doesn't fails because we don't have api. creating parser will require to change it every time we create a new schema
    
    todo:
    1. move to_next_version to a seperated file and implement it in a way that all to_next_versions are called for each new version
    2. create a test that fails if a new schema was created and no to_next_version for it was defined
    3. change new test to use local schemas and create dummy tpc for each schema version and remove test edgemdt
    4. delete implementation of _if_valid_schema 
    
    
    """

def export_target_platform_capabilities(model: schema.TargetPlatformCapabilities, export_path: Union[str, Path]) -> None:
    """
    Exports a TargetPlatformCapabilities instance to a JSON file.

    Parameters:
        model (TargetPlatformCapabilities): The TargetPlatformCapabilities instance to export.
        export_path (Union[str, Path]): The file path to export the model to.

    Raises:
        ValueError: If the model is not an instance of TargetPlatformCapabilities.
        OSError: If there is an issue writing to the file.
    """
    if not isinstance(model, schema.TargetPlatformCapabilities):
        raise ValueError("The provided model is not a valid TargetPlatformCapabilities instance.")

    path = Path(export_path)
    try:
        # Ensure the parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Export the model to JSON and write to the file
        with path.open('w', encoding='utf-8') as file:
            file.write(model.json(indent=4))
    except OSError as e:
        raise OSError(f"Failed to write to file '{export_path}': {e.strerror}") from e