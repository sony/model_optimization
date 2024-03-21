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

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.keras.back2framework.float_model_builder import FloatKerasModelBuilder
from model_compression_toolkit.core.keras.back2framework.keras_model_builder import KerasModelBuilder
from model_compression_toolkit.core.keras.back2framework.mixed_precision_model_builder import \
    MixedPrecisionKerasModelBuilder
from model_compression_toolkit.core.keras.back2framework.quantized_model_builder import QuantizedKerasModelBuilder

keras_model_builders = {ModelBuilderMode.QUANTIZED: QuantizedKerasModelBuilder,
                        ModelBuilderMode.FLOAT: FloatKerasModelBuilder,
                        ModelBuilderMode.MIXEDPRECISION: MixedPrecisionKerasModelBuilder}


def get_keras_model_builder(mode: ModelBuilderMode) -> type:
    """
    Return a Keras model builder given a ModelBuilderMode.

    Args:
        mode: Mode of the Keras model builder.

    Returns:
        Keras model builder for the given mode.
    """

    if not isinstance(mode, ModelBuilderMode):
        Logger.critical(f"Expected a ModelBuilderMode type for 'mode', but received {type(mode)} instead.")
    if mode is None:
        Logger.critical(f"get_keras_model_builder received 'mode' is None")
    if mode not in keras_model_builders.keys():
        Logger.critical(f"'mode' {mode} is not recognized in the Keras model builders factory.")
    return keras_model_builders.get(mode)
