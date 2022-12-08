# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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


from abc import abstractmethod
from typing import Any, Callable

from model_compression_toolkit.core.common import Logger


class Exporter:
    """
    Base class to define API for an Exporter class that exports and saves models.
    At initiation, it gets a model to export. This model must be an exportable model.
    Each exporter needs to implement a method called 'export' which export the model
    (convert an exportable model to a final model to run on the target platform),
    and a method named 'save_model' to allow the saving and serialization of the
    exported model.
    """

    def __init__(self, model: Any, is_layer_exportable_fn: Callable):
        """

        Args:
            model: Model to export.
            is_layer_exportable_fn: Callable to check whether a layer can be exported or not.

        """
        self.model = model
        self.is_layer_exportable_fn = is_layer_exportable_fn

    @abstractmethod
    def export(self) -> Any:
        """

        Returns: Converted model to export.

        """
        Logger.critical(f'Exporter {self.__class__} have to implement export method')

    @abstractmethod
    def save_model(self, save_model_path: str):
        """
        Save exported model in a given path.
        Args:
            save_model_path: Path to save the exported model.

        Returns:
            None.
        """
        Logger.critical(f'Exporter {self.__class__} have to implement save_model method')
