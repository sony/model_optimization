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
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformModel
import json


def load_target_platform_model(tp_model_or_path: Union[TargetPlatformModel, str]) -> TargetPlatformModel:
    """
        Parses the tp_model input, which can be either a TargetPlatformModel object
        or a string path to a JSON file.

        Parameters:
            tp_model_or_path (Union[TargetPlatformModel, str]): Input target platform model or path to .JSON file.

        Returns:
            TargetPlatformModel: The parsed TargetPlatformModel.

        Raises:
            FileNotFoundError: If the JSON file does not exist.
            ValueError: If the JSON content is invalid or cannot initialize the TargetPlatformModel.
            TypeError: If the input is neither a TargetPlatformModel nor a valid JSON file path.
        """
    if isinstance(tp_model_or_path, TargetPlatformModel):
        return tp_model_or_path

    if isinstance(tp_model_or_path, str):
        path = Path(tp_model_or_path)

        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"The path '{tp_model_or_path}' is not a valid file.")
        # Verify that the file has a .json extension
        if path.suffix.lower() != '.json':
            raise ValueError(f"The file '{path}' does not have a '.json' extension.")
        try:
            with path.open('r', encoding='utf-8') as file:
                data = file.read()
        except OSError as e:
            raise ValueError(f"Error reading the file '{tp_model_or_path}': {e.strerror}.") from e

        try:
            return TargetPlatformModel.parse_raw(data)
        except ValueError as e:
            raise ValueError(f"Invalid JSON for loading TargetPlatformModel in '{tp_model_or_path}': {e}.") from e
        except Exception as e:
            raise ValueError(f"Unexpected error while initializing TargetPlatformModel: {e}.") from e

    raise TypeError(
        f"tp_model_or_path must be either a TargetPlatformModel instance or a string path to a JSON file, "
        f"but received type '{type(tp_model_or_path).__name__}'."
    )


def export_target_platform_model(model: TargetPlatformModel, export_path: Union[str, Path]) -> None:
    """
    Exports a TargetPlatformModel instance to a JSON file.

    Parameters:
        model (TargetPlatformModel): The TargetPlatformModel instance to export.
        export_path (Union[str, Path]): The file path to export the model to.

    Raises:
        ValueError: If the model is not an instance of TargetPlatformModel.
        OSError: If there is an issue writing to the file.
    """
    if not isinstance(model, TargetPlatformModel):
        raise ValueError("The provided model is not a valid TargetPlatformModel instance.")

    path = Path(export_path)
    try:
        # Ensure the parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Export the model to JSON and write to the file
        with path.open('w', encoding='utf-8') as file:
            file.write(model.json(indent=4))
    except OSError as e:
        raise OSError(f"Failed to write to file '{export_path}': {e.strerror}") from e