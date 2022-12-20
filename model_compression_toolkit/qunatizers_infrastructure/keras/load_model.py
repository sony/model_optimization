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
from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.constants import FOUND_TF

if FOUND_TF:
    import tensorflow as tf
    from model_compression_toolkit import qunatizers_infrastructure as qi
    from model_compression_toolkit.qunatizers_infrastructure.keras.base_keras_quantizer import BaseKerasQuantizer

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
        qi_custom_objects = {subclass.__name__: subclass for subclass in BaseKerasQuantizer.__subclasses__()}
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
