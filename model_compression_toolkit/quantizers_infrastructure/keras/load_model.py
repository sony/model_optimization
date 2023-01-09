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
from typing import Set

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.constants import FOUND_TF



def _get_all_subclasses(cls: type) -> Set[type]:
    # This function returns a list of all subclasses of the given class,
    # including all subclasses of those subclasses, and so on.
    # Recursively get all subclasses of the subclass and add them to the list of all subclasses.
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in _get_all_subclasses(c)])

if FOUND_TF:
    import tensorflow as tf
    from model_compression_toolkit import quantizers_infrastructure as qi
    from model_compression_toolkit.quantizers_infrastructure import BaseKerasTrainableQuantizer
    from model_compression_toolkit.quantizers_infrastructure.keras.inferable_quantizers.base_keras_inferable_quantizer \
        import \
        BaseKerasInferableQuantizer
    keras = tf.keras

    def keras_load_quantized_model(filepath, custom_objects=None, compile=True, options=None):
        """
        This function wraps the keras load model and MCT quantization custom class to it.

        Args:
            filepath: the model file path.
            custom_objects: Additional custom objects
            compile: Boolean, whether to compile the model after loading.
            options: Optional `tf.saved_model.LoadOptions` object that specifies options for loading from SavedModel.

        Returns: A keras Model

        """
        qi_inferable_custom_objects = {subclass.__name__: subclass for subclass in _get_all_subclasses(BaseKerasInferableQuantizer)}
        qi_trainable_custom_objects = {subclass.__name__: subclass for subclass in
                                       _get_all_subclasses(BaseKerasTrainableQuantizer)}

        # Merge dictionaries into one dict
        qi_custom_objects = {**qi_inferable_custom_objects, **qi_trainable_custom_objects}

        # Add non-quantizers custom objects
        qi_custom_objects.update({qi.KerasQuantizationWrapper.__name__: qi.KerasQuantizationWrapper,
                                  qi.KerasNodeQuantizationDispatcher.__name__: qi.KerasNodeQuantizationDispatcher})
        if custom_objects is not None:
            qi_custom_objects.update(custom_objects)
        return tf.keras.models.load_model(filepath,
                                          custom_objects=qi_custom_objects, compile=compile,
                                          options=options)
else:
    def keras_load_quantized_model(filepath, custom_objects=None, compile=True, options=None):
        """
        This function wraps the keras load model and MCT quantization custom class to it.

        Args:
            filepath: the model file path.
            custom_objects: Additional custom objects
            compile: Boolean, whether to compile the model after loading.
            options: Optional `tf.saved_model.LoadOptions` object that specifies options for loading from SavedModel.

        Returns: A keras Model

        """
        Logger.critical('Installing tensorflow and tensorflow_model_optimization is mandatory '
                        'when using keras_load_quantized_model. '
                        'Could not find Tensorflow package.')
