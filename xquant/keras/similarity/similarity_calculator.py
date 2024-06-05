#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
#

from functools import partial

import keras
import logging

from mct_quantizers.keras.quantize_wrapper import KerasQuantizationWrapper

from typing import Any, Dict, Callable, List, Tuple
import numpy as np
import tensorflow as tf

from xquant.common.similarity_calculator import SimilarityCalculator


class KerasSimilarityCalculator(SimilarityCalculator):
    def __init__(self,
                 dataset_utils,
                 model_folding,
                 similarity_functions):
        super().__init__(dataset_utils,
                         model_folding,
                         similarity_functions)

    def get_activations(self,
                        float_model: keras.Model,
                        quantized_model: keras.Model,
                        float_name2quant_name,
                        data):
        def _get_activations(model: keras.Model, layer_names: List[str], data: Any) -> Tuple[Dict[str, np.ndarray], Any]:
            _model_outputs = [model.get_layer(name).output for name in layer_names] + [model.output]
            intermediate_layer_model = keras.Model(inputs=model.input, outputs=_model_outputs)
            predictions = intermediate_layer_model.predict(data)
            return {layer_name: predictions[i] for i, layer_name in enumerate(layer_names)}, predictions[-1]

        quant_activations, quant_predictions = _get_activations(quantized_model, list(float_name2quant_name.values()), data)
        float_activations, float_predictions = _get_activations(float_model, list(float_name2quant_name.keys()), data)

        if isinstance(quant_predictions, list):
            quant_predictions = np.concatenate(quant_predictions)
        if isinstance(float_predictions, list):
            float_predictions = np.concatenate(float_predictions)

        return float_activations, quant_activations, float_predictions, quant_predictions

    def get_quant_compare_points(self, quantized_model):
        return [layer.name for layer in quantized_model.layers if isinstance(layer, KerasQuantizationWrapper)]

    def get_float_candidate_layer(self,
                                  quant_compare_point,
                                  quantized_model):
        return quantized_model.get_layer(quant_compare_point).layer.name

    def get_float_layers_names(self, float_model):
        float_layers_names = [layer.name for layer in float_model.layers]
        return float_layers_names