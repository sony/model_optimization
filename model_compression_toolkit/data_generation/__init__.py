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
from model_compression_toolkit.constants import FOUND_TORCH, FOUND_TF

if FOUND_TF:
    from model_compression_toolkit.data_generation.keras.keras_data_generation import (
        tensorflow_data_generation_experimental, get_tensorflow_data_generation_config)

if FOUND_TORCH:
    from model_compression_toolkit.data_generation.pytorch.pytorch_data_generation import (
        pytorch_data_generation_experimental, get_pytorch_data_generation_config)
