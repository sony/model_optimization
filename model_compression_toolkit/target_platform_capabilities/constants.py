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

# TP Model constants
OPS_SET_LIST = 'ops_set_list'

# Version
LATEST = 'latest'


# Supported TP models names:
DEFAULT_TP_MODEL= 'default'
IMX500_TP_MODEL = 'imx500'
TFLITE_TP_MODEL = 'tflite'
QNNPACK_TP_MODEL = 'qnnpack'

# TP Attributes
KERNEL_ATTR = "kernel_attr"
BIAS_ATTR = "bias_attr"
POS_ATTR = "pos_attr"

# TODO: this is duplicated from the core frameworks constants files, because the original consts can't be used here
#  duo to circular dependency. It might be best to extract the constants from the core file and put them here (in a
#  separate changeset, because it affects the entire code)
KERAS_KERNEL = "kernel"
KERAS_DEPTHWISE_KERNEL = "depthwise_kernel"
BIAS = "bias"
PYTORCH_KERNEL = "weight"

# Configuration attributes names

WEIGHTS_N_BITS = 'weights_n_bits'
WEIGHTS_QUANTIZATION_METHOD = 'weights_quantization_method'
