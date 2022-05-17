# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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

from torch.nn import Module
from typing import Callable, List

from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.common.mixed_precision.sensitivity_evaluation_manager import \
    SensitivityEvaluationManager, compute_mp_distance_measure
from model_compression_toolkit.pytorch.back2framework.model_builder import model_builder
from model_compression_toolkit.pytorch.mixed_precision.mixed_precision_wrapper import PytorchMixedPrecisionWrapper
from model_compression_toolkit.pytorch.utils import to_torch_tensor


def get_sensitivity_evaluation(graph: Graph,
                               quant_config: MixedPrecisionQuantizationConfig,
                               metrics_weights_fn: Callable,
                               representative_data_gen: Callable,
                               fw_info: FrameworkInfo) -> Callable:
    """
    Create a function to compute the sensitivity metric of an MP model (the sensitivity
    is computed based on the similarity of the interest points' outputs between the MP model
    and the float model).
    First, we build an MP model (a model where layers that can be configured in different bitwidths use
    a PytorchMixedPrecisionWrapper) and a baseline model (a float model).
    Then, and based on the outputs of these two models (for some batches from the representative_data_gen),
    we build a function to measure the sensitivity of a change in a bitwidth of a model's layer.
    Args:
        graph: Graph to get its sensitivity evaluation for changes in bitwidths for different nodes.
        quant_config: MixedPrecisionQuantizationConfig containing parameters of how the model should be quantized.
        metrics_weights_fn: Function to compute weights for a weighted average over the distances (per layer).
        representative_data_gen: Dataset used for getting batches for inference.
        fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).
    Returns:
        Function to compute the sensitivity metric.
    """
    # Initiate a SensitivityEvaluationManager that contains all relevant objects for computing the distance metric.
    # The SensitivityEvaluationManager object initiates an MP model and a baseline model to be used for
    # comparison in the distance computation.
    # It generates and stores a set of image batches for evaluation.
    # It also runs and stores the baseline model's inference on the generated batches.
    # the model_builder method passed to the manager is the Pytorch model builder.
    sem = SensitivityEvaluationManager(graph, fw_info, quant_config, representative_data_gen, model_builder,
                                       move_tensors_func=lambda l: list(map(lambda t: t.detach().cpu(), l)))

    # Casting images tensors to torch.Tensor and putting them on same model as device
    sem.images_batches = list(map(lambda in_arr: to_torch_tensor(in_arr), sem.images_batches))

    # Initiating baseline_tensors_list since it is not initiated in SensitivityEvaluationManager init.
    sem.init_baseline_tensors_list()

    def _compute_metric(mp_model_configuration: List[int],
                        node_idx: List[int] = None) -> float:
        """
        Compute the sensitivity metric of the MP model for a given configuration (the sensitivity
        is computed based on the similarity of the interest points' outputs between the MP model
        and the float model).
        Args:
            mp_model_configuration: Bitwidth configuration to use to configure the MP model.
            node_idx: A list of nodes' indices to configure (instead of using the entire mp_model_configuration).
        Returns:
            The sensitivity metric of the MP model for a given configuration.
        """

        # Configure MP model with the given configuration.
        _configure_bitwidths_pytorch_model(sem.model_mp,
                                           sem.sorted_configurable_nodes_names,
                                           mp_model_configuration,
                                           node_idx)

        # Compute the distance matrix
        distance_matrix = sem.build_distance_metrix()

        # Configure MP model back to the same configuration as the baseline model
        baseline_mp_configuration = [0] * len(mp_model_configuration)
        _configure_bitwidths_pytorch_model(sem.model_mp,
                                           sem.sorted_configurable_nodes_names,
                                           baseline_mp_configuration,
                                           node_idx)

        return compute_mp_distance_measure(distance_matrix, metrics_weights_fn)

    return _compute_metric


def _configure_bitwidths_pytorch_model(model_mp: Module,
                                       sorted_configurable_nodes_names: List[str],
                                       mp_model_configuration: List[int],
                                       node_idx: List[int]):
    """
    Configure a dynamic Pytorch model (namely, model with layers that their weights
    bitwidth can be configured using PytorchMixedPrecisionWrapper) using an MP
    model configuration mp_model_configuration.
    Args:
        model_mp: Dynamic Pytorch model to configure. Note: model_mp is a PytorchModelBuilder object which composed of
        PytorchMixedPrecisionWrapper modules for each layer in the original module that can be configured
        for different bitwidths.
        sorted_configurable_nodes_names: List of configurable nodes names sorted topology.
        mp_model_configuration: Configuration of bitwidth indices to set to the model.
        node_idx: List of nodes' indices to configure (the rest layers are configured as the baseline model).
    """
    # Configure model
    # Note: the last configurable layer must be included in the interest points for evaluating the metric,
    # otherwise, it would not be considered throughout the mp optimization search (since it would not
    # affect the metric value)
    model_mp_layers_names = [l[0] for l in list(model_mp.named_children())]
    if node_idx is not None:  # configure specific layers in the mp model
        for node_idx_to_configure in node_idx:
            node_name = f'{sorted_configurable_nodes_names[node_idx_to_configure]}'
            if node_name in model_mp_layers_names:
                current_layer = model_mp.get_submodule(target=node_name)
                _set_layer_to_bitwidth(current_layer, mp_model_configuration[node_idx_to_configure])
            else:
                raise Exception("The last configurable node is not included in the list of interest points for"
                                "sensitivity evaluation metric for the mixed-precision search.")

    else:  # use the entire mp_model_configuration to configure the model
        for node_idx_to_configure, bitwidth_idx in enumerate(mp_model_configuration):
            node_name = f'{sorted_configurable_nodes_names[node_idx_to_configure]}'
            if node_name in model_mp_layers_names:
                current_layer = model_mp.get_submodule(target=node_name)
                _set_layer_to_bitwidth(current_layer, mp_model_configuration[node_idx_to_configure])
            else:
                raise Exception("The last configurable node is not included in the list of interest points for"
                                "sensitivity evaluation metric for the mixed-precision search.")


def _set_layer_to_bitwidth(wrapped_layer: Module,
                           bitwidth_idx: int):
    """
    Configure a layer (which is wrapped in a PytorchMixedPrecisionWrapper and holds a model's layer (nn.Module))
    to work with a different bitwidth.
    The bitwidth_idx is the index of the quantized-weights the quantizer in the PytorchMixedPrecisionWrapper holds.
    Args:
        wrapped_layer: Layer to change its bitwidth.
        bitwidth_idx: Index of the bitwidth the layer should work with.
    """
    assert isinstance(wrapped_layer, PytorchMixedPrecisionWrapper)
    # Configure the quantize_config to use a different bitwidth
    # (in practice, to use a different already quantized kernel).
    wrapped_layer.set_active_weights(bitwidth_idx)
    wrapped_layer.set_active_activation_quantizer(bitwidth_idx)
