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
from typing import Dict

from model_compression_toolkit.verify_packages import FOUND_TF
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import BaseTrainableQuantizer

if FOUND_TF:
    import tensorflow as tf
    from mct_quantizers.keras.quantize_wrapper import KerasQuantizationWrapper as InferableKerasQuantizationWrapper

    keras = tf.keras


    def _weight_name(name: str) -> str:
        """Extracts the weight name from the full TensorFlow variable name.

        For example, returns 'kernel' for 'dense_2/kernel:0'.

        Args:
          name: TensorFlow variable name.

        Returns:
          Extracted weight name.
        """
        return name.split(':')[0].split('/')[-1]


    class KerasTrainableQuantizationWrapper(InferableKerasQuantizationWrapper):

        def _set_weights_vars(self, is_training: bool = True):
            """
            This function sets weights quantizers vars to the layer.
            It's a duplicate of the function from InferableKerasQuantizationWrapper with a fix,
            that should be fixed in MCT-quantizers wrapper. Once it's fixed there, this code
            may be deleted.

            Args:
                is_training: Flag to indicate whether training or not

            Returns: None
            """
            self._weights_vars = []
            for name, quantizer in self.weights_quantizers.items():
                weight = getattr(self.layer, name)
                quantizer.initialize_quantization(weight.shape, _weight_name(weight.name) if is_training else None,
                                                  self)
                # Add weight to wrapper weight lists (rather than the layer weight lists), because it will be deleted
                # from the layer's lists after the first call
                self._weights_vars.append((name, weight, quantizer))
                if is_training and not any([weight is w for w in self._trainable_weights]):
                    self._trainable_weights.append(weight)
                elif not is_training and any([weight is w for w in self._non_trainable_weights]):
                    self._non_trainable_weights.append(weight)

        def convert_to_inferable_quantizers(self):
            """
            Convert layer's quantizers to inferable.

            Returns:
                None
            """
            # Weight quantizers
            inferable_weight_quantizers = {}
            if self.is_weights_quantization:
                for name, quantizer in self.weights_quantizers.items():
                    if hasattr(quantizer, 'convert2inferable') and callable(quantizer.convert2inferable):
                        inferable_weight_quantizers.update({name: quantizer.convert2inferable()})
                self.weights_quantizers = inferable_weight_quantizers

            # Create new layer with inferable quantizers
            inferable_quantizers_wrapper = InferableKerasQuantizationWrapper.from_config(self.get_config())
            inferable_quantizers_wrapper.layer.build(self.get_input_shape_at(0))
            weight_keys = [_weight_name(w.name) for w in inferable_quantizers_wrapper.layer.weights]
            layer_weights_list = [None] * len(weight_keys)
            for w in self.weights:
                if _weight_name(w.name) in weight_keys:
                    layer_weights_list[weight_keys.index(_weight_name(w.name))] = w
            # Verify all the weights in the list are ready. The "set_weights" method expects all the layer's weights
            if not all(w is not None for w in layer_weights_list):
                Logger.critical(f"Not all weights are set for layer '{self.layer.name}'")
            assert all(w is not None for w in layer_weights_list)
            inferable_quantizers_wrapper.set_weights(layer_weights_list)

            # The wrapper inference is using the weights of the quantizers, so it expects to create them by running _set_weights_vars
            inferable_quantizers_wrapper._set_weights_vars(False)
            inferable_quantizers_wrapper.trainable = False
            return inferable_quantizers_wrapper

else:
    class KerasTrainableQuantizationWrapper:    # pragma: no cover
        def __init__(self, *args, **kwargs):
            """
            Keras Quantization Wrapper takes a keras layer and quantizers and infer a quantized layer.

            Args:
                layer: A keras layer.
                weights_quantizers: A dictionary between a weight's name to its quantizer.
            """
            Logger.critical("Tensorflow must be installed with a version of 2.15 or lower to use "
                            "KerasTrainableQuantizationWrapper. The 'tensorflow' package is missing "
                            "or is installed with a version higher than 2.15.")
