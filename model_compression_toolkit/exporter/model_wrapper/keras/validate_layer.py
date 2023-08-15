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
from typing import Any

from mct_quantizers import BaseInferableQuantizer, KerasActivationQuantizationHolder
from model_compression_toolkit.constants import FOUND_TF
from model_compression_toolkit.logger import Logger

if FOUND_TF:
    from packaging import version
    import tensorflow as tf
    if version.parse(tf.__version__) >= version.parse("2.13"):
        from keras.src.engine.base_layer import Layer
        from keras.src.engine.input_layer import InputLayer
    else:
        from keras.engine.base_layer import Layer
        from keras.engine.input_layer import InputLayer

    from mct_quantizers import KerasQuantizationWrapper

    def is_keras_layer_exportable(layer: Any) -> bool:
        """
        Check whether a Keras layer is a valid exportable layer or not.

        Args:
            layer: Keras layer to check if considered to be valid for exporting.

        Returns:
            Check whether a Keras layer is a valid exportable layer or not.
        """
        # Keras Input layers are not wrapped
        if isinstance(layer, InputLayer):
            return True

        valid_layer = isinstance(layer, Layer)
        if not valid_layer:
            Logger.error(
                f'Exportable layer must be a Keras layer, but layer {layer.name} is of type '
                f'{type(layer)}') # pragma: no cover

        if isinstance(layer, KerasQuantizationWrapper):
            valid_weights_quantizers = isinstance(layer.weights_quantizers, dict)
            if not valid_weights_quantizers:
                Logger.error(
                    f'KerasQuantizationWrapper must have a weights_quantizers but has a '
                    f'{type(layer.weights_quantizers)} object') # pragma: no cover

            if len(layer.weights_quantizers) == 0:
                Logger.error(f'KerasQuantizationWrapper must have at least one weight quantizer, but found {len(layer.weights_quantizers)} quantizers. If layer is not quantized it should be a Keras layer.')

            for _, weights_quantizer in layer.weights_quantizers.items():
                if not isinstance(weights_quantizer, BaseInferableQuantizer):
                    Logger.error(
                        f'weights_quantizer must be a BaseInferableQuantizer object but has a '
                        f'{type(weights_quantizer)} object')  # pragma: no cover

        if isinstance(layer, KerasActivationQuantizationHolder):
            if not isinstance(layer.activation_holder_quantizer, BaseInferableQuantizer):
                Logger.error(
                    f'activation quantizer in KerasActivationQuantizationHolder'
                    f' must be a BaseInferableQuantizer object but has a '
                    f'{type(layer.activation_holder_quantizer)} object')  # pragma: no cover

        return True
else:
    def is_keras_layer_exportable(*args, **kwargs):  # pragma: no cover
        Logger.error('Installing tensorflow is mandatory '
                     'when using is_keras_layer_exportable. '
                     'Could not find Tensorflow package.')
