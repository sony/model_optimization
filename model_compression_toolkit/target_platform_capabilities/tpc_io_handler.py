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
import json
from pathlib import Path
from typing import Union

import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.target_platform_capabilities.schema.schema_compatability import is_tpc_instance, \
    tpc_to_current_schema_version, tpc_or_str_type, get_schema_by_version


def _get_json_schema(tpc_json_path: str):
    """
    Given a TPC json file path, extract the schema version from it, and return schema object matched to that
    schema version.
    """
    with open(tpc_json_path, 'r', encoding='utf-8') as f:
        schema_version = str(json.load(f)["SCHEMA_VERSION"])
    return get_schema_by_version(schema_version)


def _get_tpc_from_json(tpc_path: str) -> schema.TargetPlatformCapabilities:
    """
    Given a TPC json file path, parse it and returns a TargetPlatformCapabilities instance
    :param tpc_path: json file path
    :return: Parsed TargetPlatformCapabilities instance
    """
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
        # json_schema = _get_json_schema(tpc_path)
        # tpc = json_schema.TargetPlatformCapabilities.parse_raw(data)
        # return tpc_to_current_schema_version(tpc)
        tpc = schema.TargetPlatformCapabilities.parse_raw(data)
        return tpc_to_current_schema_version(tpc)
    except ValueError as e:
        raise ValueError(f"Invalid JSON for loading TargetPlatformCapabilities in '{tpc_path}': {e}.") from e
    except Exception as e:
        raise ValueError(f"Unexpected error while initializing TargetPlatformCapabilities: {e}.") from e


def load_target_platform_capabilities(tpc_obj_or_path: Union[tpc_or_str_type]) -> schema.TargetPlatformCapabilities:
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
    if is_tpc_instance(tpc_obj_or_path):
        tpc = tpc_obj_or_path
    elif isinstance(tpc_obj_or_path, str):
        tpc = _get_tpc_from_json(tpc_obj_or_path)
    else:
        raise TypeError(
            f"tpc_obj_or_path must be either a TargetPlatformCapabilities instance or a string path to a JSON file, "
            f"but received type '{type(tpc_obj_or_path).__name__}'."
        )

    if isinstance(tpc, schema.TargetPlatformCapabilities):  # if tpc is of current schema version
        return tpc
    return tpc_to_current_schema_version(tpc)


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
    if not is_tpc_instance(model):
        raise ValueError("The provided model is not a valid TargetPlatformCapabilities instance.")

    path = Path(export_path)
    try:
        # Ensure the parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Export the model to JSON and write to the file
        with path.open('w', encoding='utf-8') as file:
            file.write(model.model_dump_json(indent=4))
    except OSError as e:
        raise OSError(f"Failed to write to file '{export_path}': {e.strerror}") from e