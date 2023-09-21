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
from typing import Any

import mct_quantizers
from mct_quantizers.common.get_all_subclasses import get_all_subclasses

from model_compression_toolkit.constants import FOUND_TF
from model_compression_toolkit.logger import Logger

if FOUND_TF:
    import tensorflow as tf
    from tensorflow.python.saved_model.load_options import LoadOptions
    from model_compression_toolkit.trainable_infrastructure import BaseKerasTrainableQuantizer
    from model_compression_toolkit.trainable_infrastructure import KerasTrainableQuantizationWrapper
    keras = tf.keras

    def keras_load_quantized_model(filepath: str, custom_objects: Any = None, compile: bool = True,
                                   options: LoadOptions = None):
        """
        This function wraps the keras load model and adds trainable quantizers classes to its custom objects.

        Args:
            filepath: the model file path.
            custom_objects: Additional custom objects
            compile: Boolean, whether to compile the model after loading.
            options: Optional `tf.saved_model.LoadOptions` object that specifies options for loading from SavedModel.

        Returns: A keras Model

        """

        qi_trainable_custom_objects = {subclass.__name__: subclass for subclass in
                                       get_all_subclasses(BaseKerasTrainableQuantizer)}
        qi_trainable_custom_objects.update({
            KerasTrainableQuantizationWrapper.__name__: KerasTrainableQuantizationWrapper})
        all_trainable_names = list(qi_trainable_custom_objects.keys())
        if len(set(all_trainable_names)) < len(all_trainable_names):
            Logger.error(f"Found multiple quantizers with the same name that inherit from BaseKerasTrainableQuantizer"
                         f"while trying to load a model.")

        qi_custom_objects = {**qi_trainable_custom_objects}

        if custom_objects is not None:
            qi_custom_objects.update(custom_objects)
        return mct_quantizers.keras_load_quantized_model(filepath,
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
        Logger.critical('Installing tensorflow is mandatory '
                        'when using keras_load_quantized_model. '
                        'Could not find Tensorflow package.')  # pragma: no cover
