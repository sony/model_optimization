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
from model_compression_toolkit import trainable_infrastructure
from model_compression_toolkit import ptq
from model_compression_toolkit import qat
from model_compression_toolkit import exporter
from model_compression_toolkit import gptq
from model_compression_toolkit import data_generation
from model_compression_toolkit.trainable_infrastructure.keras.load_model import keras_load_quantized_model


# Old API (will not be accessible in future releases)
from model_compression_toolkit.core.common import network_editors as network_editor
from model_compression_toolkit.core.common.quantization import quantization_config
from model_compression_toolkit.core.common.mixed_precision import mixed_precision_quantization_config
from model_compression_toolkit.core.common.quantization.debug_config import DebugConfig
from model_compression_toolkit.core.common.quantization.quantization_config import QuantizationConfig, QuantizationErrorMethod, DEFAULTCONFIG
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import MixedPrecisionQuantizationConfig
from model_compression_toolkit.logger import set_log_folder
from model_compression_toolkit.core.common.data_loader import FolderImageLoader
from model_compression_toolkit.core.common.framework_info import FrameworkInfo, ChannelAxis
from model_compression_toolkit.core.common.defaultdict import DefaultDict
from model_compression_toolkit.legacy.keras_quantization_facade import keras_post_training_quantization, keras_post_training_quantization_mixed_precision
from model_compression_toolkit.legacy.pytorch_quantization_facade import pytorch_post_training_quantization, pytorch_post_training_quantization_mixed_precision
from model_compression_toolkit.core.keras.kpi_data_facade import keras_kpi_data
from model_compression_toolkit.core.pytorch.kpi_data_facade import pytorch_kpi_data
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig
from model_compression_toolkit.gptq.common.gptq_config import RoundingType
from model_compression_toolkit.gptq.keras.quantization_facade import get_keras_gptq_config
from model_compression_toolkit.gptq.pytorch.quantization_facade import get_pytorch_gptq_config

__version__ = "1.10.0"
