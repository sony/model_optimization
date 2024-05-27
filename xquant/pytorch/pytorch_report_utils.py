#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
#
from typing import Any

from mct_quantizers import PytorchQuantizationWrapper
from model_compression_toolkit.exporter.model_wrapper import get_exportable_pytorch_model
from xquant.common.edit_quantized_graph import edit_quantized_graph
from xquant.common.framework_report_utils import FrameworkReportUtils, MSE_METRIC_NAME, CS_METRIC_NAME, SQNR_METRIC_NAME
from functools import partial
import torch
import numpy as np

from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation

import model_compression_toolkit as mct


class PytorchReportUtils(FrameworkReportUtils):

    def get_edited_quantized_model(self,
                                   float_model,
                                   quantized_model,
                                   xquant_config,
                                   core_config):

        return edit_quantized_graph(quantized_graph=quantized_model.graph,
                                    fw_info=DEFAULT_PYTORCH_INFO,
                                    xquant_config=xquant_config,
                                    back2fw_fn=lambda g: get_exportable_pytorch_model(g)[0])

    def get_metric_on_output(self,
                             float_model,
                             quantized_model,
                             dataset,
                             custom_metrics_output=None,
                             is_validation=False):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dataset = partial(self.wrapped_dataset, dataset=dataset, is_validation=is_validation, device=device)

        float_model.to(device)
        quantized_model.to(device)
        float_model.eval()
        quantized_model.eval()

        metrics_to_compute = list(self.get_default_metrics().keys())
        if custom_metrics_output:
            assert isinstance(custom_metrics_output,
                              dict), f"custom_metrics_output should be a dictionary but is {type(custom_metrics_output)}"
            metrics_to_compute += list(custom_metrics_output.keys())

        metrics = {key: [] for key in metrics_to_compute}

        for x in dataset():

            with torch.no_grad():
                float_pred = float_model(*x)
                quant_pred = quantized_model(*x)
                predictions = (float_pred, quant_pred)

            results = self.compute_metrics(predictions, custom_metrics_output)
            for key in metrics:
                metrics[key].append(results[key])

        aggregated_metrics = {key: sum(value) / len(value) for key, value in metrics.items()}

        return aggregated_metrics

    def get_metric_on_intermediate(self,
                                   float_model,
                                   quantized_model,
                                   dataset,
                                   core_config: mct.core.CoreConfig,
                                   custom_metrics_intermediate=None,
                                   is_validation=False):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dataset = partial(self.wrapped_dataset, dataset=dataset, is_validation=is_validation, device=device)
        float_model = self.create_float_folded_model(float_model=float_model,
                                                     representative_dataset=dataset,
                                                     core_config=core_config)

        float_model.to(device)
        quantized_model.to(device)
        float_model.eval()
        quantized_model.eval()

        metrics_to_compute = list(self.get_default_metrics().keys())
        if custom_metrics_intermediate:
            assert isinstance(custom_metrics_intermediate,
                              dict), (f"custom_metrics_output should be a dictionary but is "
                                      f"{type(custom_metrics_intermediate)}")
            metrics_to_compute += list(custom_metrics_intermediate.keys())

        float_name2quant_name = self.get_float_to_quantized_compare_points(float_model=float_model,
                                                                           quantized_model=quantized_model)

        def get_activation(name, activations):
            def hook(model, input, output):
                activations[name] = output.detach()

            return hook

        activations_float = {}
        activations_quant = {}

        for layer_name in float_name2quant_name.keys():
            layer = dict([*float_model.named_modules()])[layer_name]
            layer.register_forward_hook(get_activation(layer_name, activations_float))

        for layer_name in float_name2quant_name.values():
            layer = dict([*quantized_model.named_modules()])[layer_name]
            layer.register_forward_hook(get_activation(layer_name, activations_quant))

        results = {}
        for float_layer, quant_layer in float_name2quant_name.items():
            results[quant_layer] = []

        for x in dataset():

            with torch.no_grad():
                _ = (float_model(*x), quantized_model(*x))

            for float_layer, quant_layer in float_name2quant_name.items():
                results[quant_layer].append(
                    self.compute_metrics((activations_float[float_layer], activations_quant[quant_layer]),
                                    custom_metrics_intermediate))

        aggregated_metrics = {}
        for layer_name, layer_metrics in results.items():
            combined_dict = {}
            for item in layer_metrics:
                for key, value in item.items():
                    if key not in combined_dict:
                        combined_dict[key] = []
                    combined_dict[key].append(value)
            for k, v in combined_dict.items():
                combined_dict[k] = np.mean(v)
            aggregated_metrics[layer_name] = combined_dict

        return aggregated_metrics

    def get_default_metrics(self):
        def compute_mse(f_pred, q_pred):
            mse = torch.nn.functional.mse_loss(f_pred, q_pred)
            return mse.item()

        def compute_cs(f_pred, q_pred):
            cs = torch.nn.functional.cosine_similarity(f_pred.flatten(),
                                                       q_pred.flatten(),
                                                       dim=0)
            return cs.item()

        def compute_sqnr(f_pred, q_pred):
            signal_power = torch.mean(f_pred ** 2)
            noise_power = torch.mean((f_pred - q_pred) ** 2)
            sqnr = signal_power / noise_power
            return sqnr.item()

        return {MSE_METRIC_NAME: compute_mse,
                CS_METRIC_NAME: compute_cs,
                SQNR_METRIC_NAME: compute_sqnr}

    def create_float_folded_model(self,
                                  float_model,
                                  representative_dataset,
                                  core_config: mct.core.CoreConfig
                                  ):
        float_graph = graph_preparation_runner(in_model=float_model,
                                               representative_data_gen=representative_dataset,
                                               fw_impl=PytorchImplementation(),
                                               fw_info=DEFAULT_PYTORCH_INFO,
                                               quantization_config=core_config.quantization_config,
                                               tpc=mct.get_target_platform_capabilities("pytorch", "default")
                                               )
        float_folded_model, _ = PytorchImplementation().model_builder(float_graph,
                                                                      mode=ModelBuilderMode.FLOAT,
                                                                      append2output=None,
                                                                      fw_info=DEFAULT_PYTORCH_INFO)
        return float_folded_model

    def wrapped_dataset(self, dataset: Any, is_validation: bool, device: str):
        """
        Generator function that wraps 'dataset' and applies device transfer.
        """
        def process_data(data, is_validation: bool, device: str):
            def transfer_to_device(_data):
                if isinstance(_data, np.ndarray):
                    return torch.from_numpy(_data).to(device)
                return _data.to(device)

            if is_validation:
                inputs = data[0]  # Assume data[0] contains the inputs and data[1] the labels
                if isinstance(inputs, list):
                    data = [transfer_to_device(t) for t in inputs]
                else:
                    data = [transfer_to_device(inputs)]
            else:
                data = [transfer_to_device(t) for t in data]

            yield data

        for x in dataset():
            return process_data(x, is_validation, device)

    def get_float_to_quantized_compare_points(self,
                                              quantized_model,
                                              float_model):

        quant_points_names = [
            n for n, m in quantized_model.named_modules()
            if isinstance(m, PytorchQuantizationWrapper)
        ]

        float_name2quant_name = {}

        for quant_point in quant_points_names:
            candidate_float_layer_name = quant_point
            if candidate_float_layer_name in [n for n, m in float_model.named_modules()]:
                if candidate_float_layer_name not in float_name2quant_name:
                    float_name2quant_name[candidate_float_layer_name] = quant_point
                else:
                    raise Exception
            else:
                print(f"Skipping point {quant_point}")

        return float_name2quant_name

    def get_quant_graph_with_metrics(self,
                                     quantized_model,
                                     collected_data,
                                     xquant_config):

        for node in quantized_model.graph.nodes:
            if xquant_config.compute_intermediate_metrics_repr:
                if node.name in collected_data["intermediate_metrics_repr"].keys():
                    node.framework_attr['xquant_repr'] = collected_data["intermediate_metrics_repr"][f"{node.name}"]
            if xquant_config.compute_intermediate_metrics_val:
                if node.name in collected_data["intermediate_metrics_val"].keys():
                    node.framework_attr['xquant_val'] = collected_data["intermediate_metrics_val"][f"{node.name}"]
        return quantized_model.graph

