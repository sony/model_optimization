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

from model_compression_toolkit.common.hardware_representation.hardware2framework import \
    FrameworkHardwareModel, LayerFilterParams
from model_compression_toolkit.common.hardware_representation.hardware2framework import \
    OperationsSetToLayers
from model_compression_toolkit.hardware_models.qnnpack import get_qnnpack_model

import tensorflow as tf
if tf.__version__ < "2.6":
    from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Conv2DTranspose, Dense, BatchNormalization, ReLU, Activation
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Conv2DTranspose, Dense, BatchNormalization, ReLU, Activation


def get_qnnpack_tensorflow():
    qnnpackhm = get_qnnpack_model()
    qnnpack_tf = FrameworkHardwareModel(qnnpackhm,
                                        name='qnnpack_tensorflow')

    with qnnpack_tf:
        OperationsSetToLayers("Conv", [Conv2D,
                                       DepthwiseConv2D,
                                       Conv2DTranspose,
                                       tf.nn.conv2d,
                                       tf.nn.depthwise_conv2d,
                                       tf.nn.conv2d_transpose])

        OperationsSetToLayers("Linear", [Dense])

        OperationsSetToLayers("BatchNorm", [BatchNormalization,
                                            tf.nn.batch_normalization])

        OperationsSetToLayers("Relu", [tf.nn.relu,
                                       tf.nn.relu6,
                                       LayerFilterParams(ReLU, negative_slope=0.0),
                                       LayerFilterParams(Activation, activation="relu")])

    return qnnpack_tf


