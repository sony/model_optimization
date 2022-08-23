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


# # Layer type constants:
PLACEHOLDER = 'placeholder'
OUTPUT = 'output'
CONSTANT = 'constant'
BUFFER = 'buffer'

# # Module operation type
CALL_FUNCTION = 'call_function'
CALL_METHOD = 'call_method'
GET_ATTR = 'get_attr'

# # Layers attributes constants:
KERNEL_SIZE = 'kernel_size'
PADDING = 'padding'
GROUPS = 'groups'
LAYER_NAME = 'name'
USE_BIAS = 'bias'
STRIDES = 'stride'
DILATIONS = 'dilation'
TENSOR_META = 'tensor_meta'
FILTERS = 'out_channels'
TYPE = 'type'
PAD = 'pad'
VALUE = 'value'
FUNCTIONAL_OP = 'functional_op'
OP_CALL_ARGS = 'op_call_args'
OP_CALL_KWARGS = 'op_call_kwargs'
INPUTS_AS_LIST = 'inputs_as_list'
INPLACE = 'inplace'
HARDTANH_MIN_VAL = 'min_val'
HARDTANH_MAX_VAL = 'max_val'

# # Layers variables names:
KERNEL = 'weight'
KERNEL_SIZE = 'kernel_size'
BIAS = 'bias'
GAMMA = 'weight'
BETA = 'bias'
IN_CHANNELS = 'in_channels'
MOVING_MEAN = 'running_mean'
MOVING_VARIANCE = 'running_var'
EPSILON = 'eps'
DIM = 'dim'
IN_CHANNELS = 'in_channels'
OUT_CHANNELS = 'out_channels'

# torch devices
CUDA = 'cuda'
CPU = 'cpu'

# ReLU bound constants
RELU_POT_BOUND = 8.0

# Supported TP models names for Pytorch:
DEFAULT_TP_MODEL = 'default'
TFLITE_TP_MODEL = 'tflite'
QNNPACK_TP_MODEL = 'qnnpack'

# MultiHeadAttention layer attributes:
EMBED_DIM = 'embed_dim'
NUM_HEADS = 'num_heads'
DROPOUT = 'dropout'
ADD_ZERO_ATTN = 'add_zero_attn'
KEY_DIM = "kdim"
VALUE_DIM = 'vdim'
BATCH_FIRST = 'batch_first'
OUT_PROJ_WEIGHT = 'out_proj.weight'
OUT_PROJ_BIAS = 'out_proj.bias'
V_PROJ_WEIGHT = 'v_proj_weight'
K_PROJ_WEIGHT = 'k_proj_weight'
Q_PROJ_WEIGHT = 'q_proj_weight'
IN_PROJ_WEIGHT = 'in_proj_weight'
IN_PROJ_BIAS = 'in_proj_bias'
BIAS_K = 'bias_k'
BIAS_V = 'bias_v'
