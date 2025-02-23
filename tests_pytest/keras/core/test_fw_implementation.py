# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, Dense, Input, Flatten
import keras

from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2keras import \
    AttachTpcToKeras


def data_gen():
    yield [np.random.randn(1, 28, 32, 10)]


def build_model():
    x = Input(shape=(28, 32, 10))
    y = Conv2D(filters=20, kernel_size=(5, 4))(x)
    y = Conv2D(filters=15, kernel_size=(4, 6), groups=5)(y)
    y = Conv2D(filters=8, kernel_size=(3, 3), strides=2)(y)
    y = Conv2D(filters=12, kernel_size=(3, 3), dilation_rate=2)(y)
    y = Conv2DTranspose(filters=20, kernel_size=(5, 3))(y)
    y = Conv2DTranspose(filters=10, kernel_size=(3, 3), strides=2)(y)
    y = Conv2DTranspose(filters=5, kernel_size=(3, 3), dilation_rate=2)(y)
    y = DepthwiseConv2D(kernel_size=(2, 3), depth_multiplier=4)(y)
    y = DepthwiseConv2D(kernel_size=(3, 3), depth_multiplier=2, strides=3)(y)
    y = DepthwiseConv2D(kernel_size=(3, 3), depth_multiplier=2, dilation_rate=2)(y)
    y = Dense(10)(y)  # 4d input
    y = Flatten()(y)
    y = Dense(5)(y)  # 2d (vector) input
    return keras.Model(inputs=x, outputs=y)


def test_get_mac(minimal_tpc):
    fw_impl = KerasImplementation()
    model = build_model()
    fw_info = DEFAULT_KERAS_INFO

    graph = graph_preparation_runner(model,
                                     data_gen,
                                     QuantizationConfig(linear_collapsing=False),
                                     fw_info=fw_info,
                                     fw_impl=fw_impl,
                                     fqc=AttachTpcToKeras().attach(minimal_tpc),
                                     mixed_precision_enable=False,
                                     running_gptq=False)

    nodes = graph.get_topo_sorted_nodes()
    assert len(nodes) == 14, nodes
    assert fw_impl.get_node_mac_operations(nodes[0], fw_info) == 0
    assert fw_impl.get_node_mac_operations(nodes[1], fw_info) == (10*20*5*4)*24*29
    assert fw_impl.get_node_mac_operations(nodes[2], fw_info) == (4*3*4*6)*5*21*24
    assert fw_impl.get_node_mac_operations(nodes[3], fw_info) == (15*8*3*3)*10*11
    assert fw_impl.get_node_mac_operations(nodes[4], fw_info) == (8*12*3*3)*6*7
    assert fw_impl.get_node_mac_operations(nodes[5], fw_info) == (12*20*5*3)*10*9
    assert fw_impl.get_node_mac_operations(nodes[6], fw_info) == (20*10*3*3)*21*19
    assert fw_impl.get_node_mac_operations(nodes[7], fw_info) == (10*5*3*3)*25*23
    assert fw_impl.get_node_mac_operations(nodes[8], fw_info) == (5*2*3*4)*24*21
    assert fw_impl.get_node_mac_operations(nodes[9], fw_info) == (10*3*3*4)*8*7
    assert fw_impl.get_node_mac_operations(nodes[10], fw_info) == (40*3*3*2)*4*3
    assert fw_impl.get_node_mac_operations(nodes[11], fw_info) == 4*3*(80*10)
    assert fw_impl.get_node_mac_operations(nodes[12], fw_info) == 0
    assert fw_impl.get_node_mac_operations(nodes[13], fw_info) == (4*3*10)*5
