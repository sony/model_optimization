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


# Activation functions strings:
SOFTMAX = 'softmax'
SIGMOID = 'sigmoid'
LINEAR = 'linear'
IDENTITY = 'identity'
TANH = 'tanh'
SWISH = 'swish'
RELU = 'relu'
SELU = 'selu'
GELU = 'gelu'
ARGMAX = 'argmax'

# Layers attributes constants:
KERNEL_SIZE = 'kernel_size'
PADDING = 'padding'
GROUPS = 'groups'
STRIDES = 'strides'
DILATIONS = 'dilation_rate'
DATA_FORMAT = 'data_format'
LAYER_NAME = 'name'
TRAINABLE = 'trainable'
ACTIVATION = 'activation'
USE_BIAS = 'use_bias'
FILTERS = 'filters'
UNITS = 'units'
PAD_VALID = 'valid'
PAD_SAME = 'same'
RELU_MAX_VALUE = 'max_value'
THRESHOLD = 'threshold'
NEGATIVE_SLOPE = 'negative_slope'
CHANNELS_FORMAT = 'data_format'
CHANNELS_FORMAT_FIRST = 'channels_first'
CHANNELS_FORMAT_LAST = 'channels_last'
AXES = 'axes'
AXIS = 'axis'
DIMS = 'dims'
TARGET_SHAPE = 'target_shape'
TRANSPOSE_A = 'transpose_a'
TRANSPOSE_B = 'transpose_b'

# functional nodes attributes
FUNCTION = 'function'
F_RESHAPE = 'reshape'
F_STRIDED_SLICE = 'strided_slice'
F_MATMUL = 'matmul'
F_STACK = 'stack'
F_STRIDED_SLICE_BEGIN = 'begin_mask'
F_STRIDED_SLICE_END = 'end_mask'

# Layers variables names:
KERNEL = 'kernel'
DEPTHWISE_KERNEL = 'depthwise_kernel'
BIAS = 'bias'
GAMMA = 'gamma'
BETA = 'beta'
CENTER = 'center'
SCALE = 'scale'
MOVING_MEAN = 'moving_mean'
MOVING_VARIANCE = 'moving_variance'
EPSILON = 'epsilon'
EPSILON_VAL = 1e-5
MOMENTUM = 'momentum'
MOMENTUM_VAL = 0.9

# MultiHeadAttention layer attributes:
NUM_HEADS = 'num_heads'
KEY_DIM = 'key_dim'
VALUE_DIM = 'value_dim'
QUERY_SHAPE = 'query_shape'
KEY_SHAPE = 'key_shape'
VALUE_SHAPE = 'value_shape'
OUTPUT_SHAPE = 'output_shape'
ATTENTION_AXES = 'attention_axes'
Q_KERNEL = '/query/kernel'
K_KERNEL = '/key/kernel'
V_KERNEL = '/value/kernel'
Q_BIAS = '/query/bias'
K_BIAS = '/key/bias'
V_BIAS = '/value/bias'
OUTPUT_KERNEL = '/attention_output/kernel'
OUTPUT_BIAS = '/attention_output/bias'

# ReLU bound constants
RELU_POT_BOUND = 8.0

# Supported TP models names for Tensorflow:
DEFAULT_TP_MODEL = 'default'
TFLITE_TP_MODEL = 'tflite'
QNNPACK_TP_MODEL = 'qnnpack'


# TFOpLambda functions:
ADD = 'add'
PAD = 'pad'
