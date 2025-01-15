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
from typing import Union

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformCapabilities
import json


def load_target_platform_capabilities(tpc_obj_or_path: Union[TargetPlatformCapabilities, str]) -> TargetPlatformCapabilities:
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
    if isinstance(tpc_obj_or_path, TargetPlatformCapabilities):
        return tpc_obj_or_path

    if isinstance(tpc_obj_or_path, str):
        path = Path(tpc_obj_or_path)

        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"The path '{tpc_obj_or_path}' is not a valid file.")
        # Verify that the file has a .json extension
        if path.suffix.lower() != '.json':
            raise ValueError(f"The file '{path}' does not have a '.json' extension.")
        try:
            with path.open('r', encoding='utf-8') as file:
                data = file.read()
        except OSError as e:
            raise ValueError(f"Error reading the file '{tpc_obj_or_path}': {e.strerror}.") from e

        try:
            return TargetPlatformCapabilities.parse_raw(data)
        except ValueError as e:
            raise ValueError(f"Invalid JSON for loading TargetPlatformCapabilities in '{tpc_obj_or_path}': {e}.") from e
        except Exception as e:
            raise ValueError(f"Unexpected error while initializing TargetPlatformCapabilities: {e}.") from e

    raise TypeError(
        f"tpc_obj_or_path must be either a TargetPlatformCapabilities instance or a string path to a JSON file, "
        f"but received type '{type(tpc_obj_or_path).__name__}'."
    )


def export_target_platform_capabilities(model: TargetPlatformCapabilities, export_path: Union[str, Path]) -> None:
    """
    Exports a TargetPlatformCapabilities instance to a JSON file.

    Parameters:
        model (TargetPlatformCapabilities): The TargetPlatformCapabilities instance to export.
        export_path (Union[str, Path]): The file path to export the model to.

    Raises:
        ValueError: If the model is not an instance of TargetPlatformCapabilities.
        OSError: If there is an issue writing to the file.
    """
    if not isinstance(model, TargetPlatformCapabilities):
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