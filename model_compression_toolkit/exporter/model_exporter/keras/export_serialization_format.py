# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from enum import Enum


class KerasExportSerializationFormat(Enum):
    """
    Specify which serialization format to use for exporting a quantized Keras model.

    KERAS_H5 - .keras (TF2.13 and above) or .h5 (TF2.12 and below) file format

    TFLITE - .tflite file format

    """

    KERAS_H5 = 0
    TFLITE = 1
