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

from typing import Dict, Tuple, Any

from keras.engine.base_layer import Layer
from keras.layers import TFOpLambda
from tensorflow import TensorShape
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper


class ExtendedQuantizeWrapper(QuantizeWrapper):

    """Quantizes the weights and activations of the keras layer it wraps, according
    to the quantization config that is passed. This class was created to deal with TFOpLambda that can
    not use TF QuantizeWrapper since it does not implement compute_output_shape.
    Notice that reused layers do not have a compute_output_shape method, thus the added method
    is irrelevant for wrapping them."""

    def __init__(self,
                 layer:Layer,
                 quantize_config:QuantizeConfig,
                 output_shape:Tuple[Any]=None,
                 **kwargs:Dict[str, Any]):
        """
        Create a wrapper for a keras layer.

        Args:
            layer: The keras layer to be quantized.
            quantize_config: `QuantizeConfig` to quantize the layer.
            output_shape: The output shape of the layer.
            **kwargs: Keyword arguments to build the base class QuantizeWrapper.
        """

        # TFOpLambda does not implement the method compute_output_shape which is mandatory for cloning the model
        # and use TF transformations. For this reason, we add the output_shape in the layer configuration and
        # add an implementation for compute_output_shape.
        self._output_shape = output_shape
        if isinstance(layer, TFOpLambda):
            layer.compute_output_shape = self._compute_output_shape

        super(ExtendedQuantizeWrapper, self).__init__(layer=layer,
                                                      quantize_config=quantize_config,
                                                      **kwargs)

    def _compute_output_shape(self, input_shape: TensorShape) -> Tuple[Any]:
        """
        Internal method that returns the output shape of the layer.

        Args:
            input_shape: Input shape the layer is expecting to have.

        Returns:
            The layer's output shape.
        """

        return self._output_shape

    def get_config(self) -> Dict[str, Any]:
        """

        Returns: The layer configuration with the output shape, so it can be deserialized.

        """
        cfg = super(ExtendedQuantizeWrapper, self).get_config()
        cfg.update({'output_shape': self._output_shape})
        return cfg

