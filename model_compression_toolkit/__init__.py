# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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


from model_compression_toolkit.target_platform_capabilities import target_platform
from model_compression_toolkit.target_platform_capabilities.tpc_models.get_target_platform_capabilities import get_target_platform_capabilities
from model_compression_toolkit import core
from model_compression_toolkit.logger import set_log_folder
from model_compression_toolkit.legacy.keras_quantization_facade import keras_post_training_quantization, keras_post_training_quantization_mixed_precision
from model_compression_toolkit.legacy.pytorch_quantization_facade import pytorch_post_training_quantization, pytorch_post_training_quantization_mixed_precision
from model_compression_toolkit import quantizers_infrastructure
from model_compression_toolkit import ptq
from model_compression_toolkit import qat
from model_compression_toolkit import exporter
from model_compression_toolkit import gptq
from model_compression_toolkit.gptq import GradientPTQConfig


__version__ = "1.8.0"
