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

from functools import partial
from typing import Dict, Callable
from mct_quantizers.pytorch.quantize_wrapper import PytorchQuantizationWrapper
import torch

from model_compression_toolkit.core.pytorch.utils import set_model, get_working_device
from xquant.common.similarity_calculator import SimilarityCalculator


class PytorchSimilarityCalculator(SimilarityCalculator):
    def __init__(self,
                 dataset_utils,
                 model_folding,
                 similarity_functions):
        super().__init__(dataset_utils,
                         model_folding,
                         similarity_functions,
                         device=get_working_device())

    def get_activations(self,
                        float_model: torch.nn.Module,
                        quantized_model: torch.nn.Module,
                        float_name2quant_name,
                        data):
        def _get_activation(name: str, activations: dict):
            def hook(model, input, output):
                activations[name] = output.detach()

            return hook

        activations_float = {}
        activations_quant = {}

        for layer_name in float_name2quant_name.keys():
            layer = dict([*float_model.named_modules()])[layer_name]
            layer.register_forward_hook(_get_activation(layer_name, activations_float))

        for layer_name in float_name2quant_name.values():
            layer = dict([*quantized_model.named_modules()])[layer_name]
            layer.register_forward_hook(_get_activation(layer_name, activations_quant))

        with torch.no_grad():
            float_predictions = float_model(*data)
            quant_predictions = quantized_model(*data)

        return activations_float, activations_quant, float_predictions, quant_predictions

    def get_quant_compare_points(self, quantized_model:torch.nn.Module):
        return [n for n, m in quantized_model.named_modules() if isinstance(m, PytorchQuantizationWrapper)]

    def get_float_candidate_layer(self, quant_compare_point, quantized_model):
        return quant_compare_point

    def get_float_layers_names(self, float_model):
        return [n for n, m in float_model.named_modules()]


