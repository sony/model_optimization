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
import keras

from mct_quantizers import KerasQuantizationWrapper

if keras.__version__ >= "2.13":
    from keras.src.layers import TFOpLambda
else:
    from keras.layers import TFOpLambda


def get_layers_from_model_by_type(model:keras.Model,
                                  layer_type: type,
                                  include_wrapped_layers: bool = True):
    """
    Return a list of layers of some type from a Keras model. The order of the returned list
    is according the order of the layers in model.layers.
    If include_wrapped_layers is True, layers from that type that are wrapped using KerasQuantizationWrapper
    are returned as well.

    Args:
        model: Keras model to get its layers.
        layer_type: Type of the layer we want to retrieve from the model.
        include_wrapped_layers: Whether or not to include layers that are wrapped using KerasQuantizationWrapper.

    Returns:
        List of layers from type layer_type from the model.
    """

    match_type = lambda l, t: type(l) == t or (isinstance(l, TFOpLambda) and l.symbol == TFOpLambda(t).symbol)

    return [layer for layer in model.layers if match_type(layer, layer_type) or
            include_wrapped_layers and isinstance(layer, KerasQuantizationWrapper) and match_type(layer.layer, layer_type)]
