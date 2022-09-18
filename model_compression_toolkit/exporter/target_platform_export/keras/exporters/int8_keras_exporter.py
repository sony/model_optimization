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
from model_compression_toolkit.exporter.target_platform_export.keras.exporters.base_keras_exporter import BaseKerasExporter
import keras

class Int8KerasExporter(BaseKerasExporter):
    """
    Exporter for int8 Keras models.
    """

    def export(self) -> keras.models.Model:
        """
        Convert fully-quantized Keras model to an int8 export-ready model.

        Returns:
            Int8 Keras model ready to export.
        """
        pass

    def save_model(self, save_model_path: str):
        """
        Save exported model to a given path.
        Args:
            save_model_path: Path to save the model.

        Returns:
            None.
        """
        pass
