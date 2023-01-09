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

import tensorflow as tf
from tensorflow.keras.layers import Layer

from model_compression_toolkit import quantizers_infrastructure as qi
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.keras.back2framework.keras_model_builder import KerasModelBuilder
from model_compression_toolkit.exporter.model_wrapper.keras.builder.node_to_dispatcher import \
    get_quantization_dispatcher


def _get_wrapper(node: common.BaseNode,
                 layer: Layer) -> qi.KerasQuantizationWrapper:
    """
    A function which takes a computational graph node and a keras layer and perform the quantization wrapping
    Args:
        n: A node of mct graph.
        layer: A keras layer

    Returns: Wrapped layer

    """
    return qi.KerasQuantizationWrapper(layer,
                                       get_quantization_dispatcher(node))


def get_exportable_keras_model(graph: Graph) -> tf.keras.models.Model:
    """
    Convert graph to an exportable Keras model (model with all quantization parameters).
    An exportable model can then be exported using model_exporter, to retrieve the
    final exported model.

    Args:
        graph: Graph to convert to an exportable Keras model.

    Returns:
        Exportable Keras model.
    """
    return KerasModelBuilder(graph=graph,
                             wrapper=_get_wrapper).build_model()
