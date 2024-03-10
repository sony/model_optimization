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
from model_compression_toolkit.qat.common.qat_config import QATConfig, TrainingMethod

from model_compression_toolkit.qat.keras.quantization_facade import keras_quantization_aware_training_init_experimental, keras_quantization_aware_training_finalize_experimental
from model_compression_toolkit.qat.pytorch.quantization_facade import pytorch_quantization_aware_training_init_experimental, pytorch_quantization_aware_training_finalize_experimental