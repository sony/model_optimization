# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
from model_compression_toolkit.core.tpc_models.default_tpc.v3.tp_model import get_tp_model, generate_tp_model, get_op_quantization_configs
from model_compression_toolkit.core.tpc_models.default_tpc.v3.tpc_keras import get_keras_tpc, generate_keras_tpc
from model_compression_toolkit.core.tpc_models.default_tpc.v3.tpc_pytorch import get_pytorch_tpc, generate_pytorch_tpc
