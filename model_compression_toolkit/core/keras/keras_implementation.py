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
from typing import List, Any, Tuple, Callable, Type, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.layers.base import Layer

from model_compression_toolkit.core.common.mixed_precision.sensitivity_evaluation import SensitivityEvaluation
from model_compression_toolkit.core.common.similarity_analyzer import compute_kl_divergence, compute_cs, compute_mse
from model_compression_toolkit.core.keras.back2framework.model_gradients import \
    keras_iterative_approx_jacobian_trace
from model_compression_toolkit.core.keras.constants import ACTIVATION, SOFTMAX, SIGMOID, ARGMAX, LAYER_NAME
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.batchnorm_reconstruction import \
    keras_batchnorm_reconstruction
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.virtual_activation_weights_composition import \
    VirtualActivationWeightsComposition
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.weights_activation_split import \
    WeightsActivationSplit
from model_compression_toolkit.core.keras.mixed_precision.set_layer_to_bitwidth import set_layer_to_bitwidth
from model_compression_toolkit.core.keras.statistics_correction.apply_second_moment_correction import \
    keras_apply_second_moment_correction
from packaging import version

if version.parse(tf.__version__) < version.parse("2.6"):
    from tensorflow.keras.layers import Dense, Activation, Conv2D, DepthwiseConv2D, Conv2DTranspose, Concatenate, Add
    from tensorflow.python.keras.layers.core import TFOpLambda
else:
    from keras.layers import Dense, Activation, Conv2D, DepthwiseConv2D, Conv2DTranspose, \
        Concatenate, Add
    from keras.layers.core import TFOpLambda

from model_compression_toolkit import QuantizationConfig, FrameworkInfo, CoreConfig, MixedPrecisionQuantizationConfigV2
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.collectors.statistics_collector import BaseStatsCollector
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.common.node_prior_info import NodePriorInfo
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.gptq.common.gptq_training import GPTQTrainer
from model_compression_toolkit.gptq.keras.gptq_training import KerasGPTQTrainer
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.activation_decomposition import \
    ActivationDecomposition
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.softmax_shift import \
    keras_softmax_shift
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.batchnorm_folding import \
    keras_batchnorm_folding
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.batchnorm_refusing import \
    keras_batchnorm_refusing
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.linear_collapsing import \
    keras_linear_collapsing
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.residual_collapsing import \
    keras_residual_collapsing
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.input_scaling import InputScaling, \
    InputScalingWithPad
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.relu_bound_to_power_of_2 import \
    ReLUBoundToPowerOfTwo
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.remove_relu_upper_bound import \
    RemoveReLUUpperBound
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.multi_head_attention_decomposition import \
    MultiHeadAttentionDecomposition
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.scale_equalization import \
    ScaleEqualization, ScaleEqualizationWithPad, ScaleEqualizationMidActivation, ScaleEqualizationMidActivationWithPad
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.separableconv_decomposition import \
    SeparableConvDecomposition, DEPTH_MULTIPLIER
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.shift_negative_activation import \
    keras_apply_shift_negative_correction
from model_compression_toolkit.core.keras.keras_node_prior_info import create_node_prior_info
from model_compression_toolkit.core.keras.reader.reader import model_reader
from model_compression_toolkit.core.common.collectors.statistics_collector_generator import \
    create_stats_collector_for_node
import model_compression_toolkit.core.keras.constants as keras_constants
from model_compression_toolkit.core.keras.tf_tensor_numpy import tf_tensor_to_numpy, to_tf_tensor
from model_compression_toolkit.core.keras.back2framework import get_keras_model_builder


class KerasImplementation(FrameworkImplementation):
    """
    An class with implemented methods to support optimizing Keras models.
    """

    def __init__(self):
        super().__init__()

    @property
    def constants(self):
        """

        Returns: Module of Keras constants.

        """
        return keras_constants

    def model_reader(self,
                     model: Model,
                     representative_data_gen: Callable) -> Graph:
        """
        Convert a framework's model into a graph.
        Args:
            model: Framework's model.
            representative_data_gen (Callable): Dataset used for calibration.

        Returns:
            Graph representing the input model.
        """
        return model_reader(model)

    def to_numpy(self, tensor: tf.Tensor) -> np.ndarray:
        """
        Convert framework's tensor to a Numpy array.
        Args:
            tensor: Framework's tensor.

        Returns:
            Numpy array converted from the input tensor.
        """
        return tf_tensor_to_numpy(tensor)

    def to_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """
        Convert a Numpy array to a framework's tensor.
        Args:
            tensor: Numpy array.

        Returns:
            Framework's tensor converted from the input Numpy array.
        """
        return to_tf_tensor(tensor)

    def model_builder(self,
                      graph: Graph,
                      mode: ModelBuilderMode,
                      append2output: List[Any] = None,
                      fw_info: FrameworkInfo = DEFAULT_KERAS_INFO,
                      return_float_outputs: bool = False) -> Tuple[Model, UserInformation]:
        """
        Build a Keras model from a graph.
        The mode determines how the model should be build. append2output is a list of Nodes
        to set as the model outputs.

        Args:
            graph: Graph to build the model from it.
            mode: Mode for how to build the model.
            append2output: List of Nodes to set as the model's outputs.
            fw_info: FrameworkInfo object with information about the specific framework's model
            return_float_outputs (bool): whether to return outputs before or after quantization nodes (default)
        Returns:
            A tuple of the Keras model that was built and an UserInformation object.
        """

        keras_model_builder = get_keras_model_builder(mode)
        return keras_model_builder(graph=graph,
                                   append2output=append2output,
                                   fw_info=fw_info,
                                   return_float_outputs=return_float_outputs).build_model()

    def run_model_inference(self,
                            model: Any,
                            input_list: List[Any]) -> Tuple[tf.Tensor]:
        """
        Run the model logic on the given the inputs.

        Args:
            model: Keras model.
            input_list: List of inputs for the model.

        Returns:
            The Keras model's output.
        """
        return model(input_list)

    def shift_negative_correction(self,
                                  graph: Graph,
                                  core_config: CoreConfig,
                                  fw_info: FrameworkInfo) -> Graph:
        """
        Apply shift negative correction (SNC) on a graph.

        Args:
            graph: Graph to apply SNC on.
            core_config: Quantization configuration.
            fw_info: FrameworkInfo object with information about the specific framework's model.

        Returns:
            Graph after SNC.
        """
        return keras_apply_shift_negative_correction(graph,
                                                     core_config,
                                                     fw_info)

    def attach_sc_to_node(self,
                          node: BaseNode,
                          fw_info: FrameworkInfo) -> BaseStatsCollector:
        """
        Return a statistics collector that should be attached to a node's output
        during statistics collection.

        Args:
            node: Node to return its collector.
            fw_info: Information relevant to a specific framework about what is out channel axis (for statistics per-channel)

        Returns:
            Statistics collector for the node.
        """
        return create_stats_collector_for_node(node, fw_info)

    def get_substitutions_channel_equalization(self,
                                               quant_config: QuantizationConfig,
                                               fw_info: FrameworkInfo) -> List[common.BaseSubstitution]:
        """
        Return a list of the framework substitutions used for channel equalization.

        Args:
            quant_config: QuantizationConfig to determine which substitutions to return.
            fw_info: FrameworkInfo object with information about the specific framework's model.

        Returns:
            A list of the framework substitutions used after we collect statistics.
        """
        substitutions_list = []
        if quant_config.activation_channel_equalization:
            substitutions_list.extend([ScaleEqualization(quant_config, fw_info),
                                       ScaleEqualizationWithPad(quant_config, fw_info),
                                       ScaleEqualizationMidActivation(quant_config, fw_info),
                                       ScaleEqualizationMidActivationWithPad(quant_config, fw_info)])
        return substitutions_list

    def get_substitutions_prepare_graph(self, fw_info: FrameworkInfo = None) -> List[common.BaseSubstitution]:
        """

        Returns: A list of the framework substitutions used to prepare the graph.

        """
        return [SeparableConvDecomposition(),
                MultiHeadAttentionDecomposition(),
                ActivationDecomposition()]

    def get_substitutions_pre_statistics_collection(self, quant_config: QuantizationConfig) -> \
            List[common.BaseSubstitution]:
        """
        Return a list of the framework substitutions used before we collect statistics.

        Args:
            quant_config: QuantizationConfig to determine which substitutions to return.

        Returns:
            A list of the framework substitutions used before we collect statistics.

        """
        substitutions_list = [keras_batchnorm_folding()]
        if quant_config.relu_bound_to_power_of_2:
            substitutions_list.append(ReLUBoundToPowerOfTwo())
        return substitutions_list

    def get_substitutions_statistics_correction(self, quant_config: QuantizationConfig) -> \
            List[common.BaseSubstitution]:
        """
        Returns A list of the framework substitutions used for statistics correction.

        Args:
            quant_config: QuantizationConfig to determine which substitutions to return.

        Returns:
            A list of the framework substitutions used for statistics correction.
        """
        substitutions_list = []
        if quant_config.weights_second_moment_correction:
            substitutions_list.append(keras_batchnorm_reconstruction())
        return substitutions_list

    def get_residual_collapsing_substitution(self) -> List[common.BaseSubstitution]:
        """
        Returns: A list of the framework substitutions used for residual collapsing
        """
        substitutions_list = [keras_residual_collapsing()]
        return substitutions_list

    def get_linear_collapsing_substitution(self) -> common.BaseSubstitution:
        """
        Returns: linear collapsing substitution
        """
        return keras_linear_collapsing()

    def get_substitutions_post_statistics_collection(self, quant_config: QuantizationConfig) \
            -> List[common.BaseSubstitution]:
        """
        Return a list of the framework substitutions used after we collect statistics.

        Args:
            quant_config: QuantizationConfig to determine which substitutions to return.

        Returns:
            A list of the framework substitutions used after we collect statistics.
        """
        substitutions_list = []
        if quant_config.softmax_shift:
            substitutions_list.append(keras_softmax_shift())
        if quant_config.input_scaling:
            substitutions_list.append(InputScaling())
            substitutions_list.append(InputScalingWithPad())
        return substitutions_list

    def get_substitutions_pre_build(self) -> List[common.BaseSubstitution]:
        """

        Returns: A list of the framework substitutions used before we build a quantized model.

        """

        return [RemoveReLUUpperBound()]

    def get_substitutions_virtual_weights_activation_coupling(self) -> List[common.BaseSubstitution]:
        """
        Returns: A list of Keras substitutions used to build a virtual graph with composed activation-weights pairs.
        """

        return [WeightsActivationSplit(),
                VirtualActivationWeightsComposition()]

    def get_substitutions_after_second_moment_correction(self, quant_config: QuantizationConfig) \
            -> List[common.BaseSubstitution]:
        """
        Return a list of the framework substitutions used after second moment statistics.

        Args:
            quant_config: QuantizationConfig to determine which substitutions to return.

        Returns:
            A list of the framework substitutions used after we apply second moment statistics.
        """
        substitutions_list = []
        if quant_config.weights_second_moment_correction:
            substitutions_list.append(keras_batchnorm_refusing())
        return substitutions_list

    def get_gptq_trainer_obj(self) -> Type[GPTQTrainer]:
        """
        Returns:  Keras object of GPTQTrainer
        """
        return KerasGPTQTrainer

    def get_sensitivity_evaluator(self,
                                  graph: Graph,
                                  quant_config: MixedPrecisionQuantizationConfigV2,
                                  representative_data_gen: Callable,
                                  fw_info: FrameworkInfo,
                                  disable_activation_for_metric: bool = False) -> SensitivityEvaluation:
        """
        Creates and returns an object which handles the computation of a sensitivity metric for a mixed-precision
        configuration (comparing to the float model).

        Args:
            graph: Graph to build its float and mixed-precision models.
            quant_config: QuantizationConfig of how the model should be quantized.
            representative_data_gen: Dataset to use for retrieving images for the models inputs.
            fw_info: FrameworkInfo object with information about the specific framework's model.
            disable_activation_for_metric: Whether to disable activation quantization when computing the MP metric.

        Returns:
            A SensitivityEvaluation object.
        """

        return SensitivityEvaluation(graph=graph,
                                     quant_config=quant_config,
                                     representative_data_gen=representative_data_gen,
                                     fw_info=fw_info,
                                     fw_impl=self,
                                     set_layer_to_bitwidth=set_layer_to_bitwidth,
                                     get_quant_node_name=lambda node_name: f'quant_{node_name}',
                                     disable_activation_for_metric=disable_activation_for_metric)

    def get_node_prior_info(self,
                            node: BaseNode,
                            fw_info: FrameworkInfo,
                            graph: Graph) -> NodePriorInfo:
        """
        Get a NodePriorInfo object for a node that represents a Keras layer.

        Args:
            node: Node to get its prior info.
            fw_info: Framework specific information needed to create the prior info of the node.
            graph: Graph to check the next node type.

        Returns:
            NodePriorInfo with information about the node.
        """

        return create_node_prior_info(node=node,
                                      fw_info=fw_info, graph=graph)

    def count_node_for_mixed_precision_interest_points(self, node: BaseNode) -> bool:
        """
        Returns whether a given node in considered as a potential interest point for mp metric computation purposes.
        Args:
            node: Node to indicate whether it needs to be part of the interest points set.
        Returns: True if the node should be considered an interest point, False otherwise.
        """

        if node.type == Activation:
            node_type_name = node.framework_attr[keras_constants.ACTIVATION]
            if node_type_name in [keras_constants.SOFTMAX, keras_constants.SIGMOID]:
                return True
        elif node.type in [tf.nn.softmax, tf.nn.sigmoid, Conv2D, DepthwiseConv2D, Conv2DTranspose, Dense, Concatenate,
                           Add, tf.add]:
            return True

        return False

    def get_node_distance_fn(self, layer_class: type,
                             framework_attrs: Dict[str, Any],
                             compute_distance_fn: Callable = None) -> Callable:
        """
        A mapping between layers' types and a distance function for computing the distance between
        two tensors (for loss computation purposes). Returns a specific function if node of specific types is
        given, or a default (normalized MSE) function otherwise.

        Args:
            layer_class: Class path of a model's layer.
            framework_attrs: Framework attributes the layer had which the graph node holds.
            compute_distance_fn: An optional distance function to use globally for all nodes.

        Returns: A distance function between two tensors.
        """

        if compute_distance_fn is not None:
            return compute_distance_fn

        if layer_class == Activation:
            node_type_name = framework_attrs[ACTIVATION]
            if node_type_name == SOFTMAX:
                return compute_kl_divergence
            elif node_type_name == SIGMOID:
                return compute_cs
        elif layer_class == tf.nn.softmax:
            return compute_kl_divergence
        elif layer_class == tf.nn.sigmoid:
            return compute_cs
        elif layer_class == Dense:
            return compute_cs
        return compute_mse

    def get_model_layers_names(self,
                               model: Model) -> List[str]:
        """
        Returns a list of the given model's layers names.

        Args:
            model: A Keras model.

        Returns: List of layers' names.

        """

        return [layer.name for layer in model.layers]

    def get_model_layer_by_name(self,
                                model: Model,
                                layer_name: str) -> Layer:
        """
        Returns a Keras model's layer by its name.

        Args:
            model: A Keras model to retrieve a layer from.
            layer_name: The requested layer's name.

        Returns: A Keras layer object.

        """

        return model.get_layer(name=layer_name)

    def model_grad(self,
                   graph_float: common.Graph,
                   model_input_tensors: Dict[BaseNode, np.ndarray],
                   interest_points: List[BaseNode],
                   output_list: List[BaseNode],
                   all_outputs_indices: List[int],
                   alpha: float = 0.3,
                   n_iter: int = 50,
                   norm_weights: bool = True) -> List[float]:
        """
        Calls a Keras model gradient calculation function, which computes the jacobian-based weights of the model's
        outputs with respect to the feature maps of the set of given interest points.

        Args:
            graph_float: Graph to build its corresponding Keras model.
            model_input_tensors: A mapping between model input nodes to an input batch.
            interest_points: List of nodes which we want to get their feature map as output, to calculate distance metric.
            output_list: List of nodes that considered as model's output for the purpose of gradients computation.
            all_outputs_indices: Indices of the model outputs and outputs replacements (if exists),
                in a topological sorted interest points list
            alpha:A tuning parameter to allow calibration between the contribution of the output feature maps returned
                weights and the other feature maps weights (since the gradient of the output layers does not provide a
                compatible weight for the distance metric computation).
            n_iter: The number of random iterations to calculate the approximated  jacobian-based weights for each interest point.
            norm_weights: Whether to normalize the returned weights (to get values between 0 and 1).

        Returns: A list of (possibly normalized) jacobian-based weights to be considered as the relevancy that each interest
        point's output has on the model's output.

        """

        return keras_iterative_approx_jacobian_trace(graph_float, model_input_tensors, interest_points, output_list,
                                                     all_outputs_indices, alpha, n_iter, norm_weights=norm_weights)

    def is_node_compatible_for_metric_outputs(self,
                                              node: BaseNode) -> Any:
        """
        Checks and returns whether the given node is compatible as output for metric computation
        purposes and gradient-based weights calculation.

        Args:
            node: A BaseNode object.

        Returns: Whether the node is compatible as output for metric computation or not.

        """

        if node.layer_class == TFOpLambda:
            node_attr = getattr(node, 'framework_attr', None)
            if node_attr is not None and ARGMAX in node_attr[LAYER_NAME]:
                return False
        elif node.layer_class == tf.nn.softmax or node.layer_class == tf.math.argmax:
            return False

        return True

    def get_node_mac_operations(self,
                                node: BaseNode,
                                fw_info: FrameworkInfo) -> float:
        """
        Gets the MAC operation count for a given operation.

        Args:
            node: A graph node that wraps the operation for which the MAC count is computed.
            fw_info: FrameworkInfo object with information about the Keras model.

        Returns: The MAC count og the operation
        """

        input_shape = node.input_shape
        output_shape = node.output_shape
        kernel_shape = node.get_weights_by_keys(fw_info.get_kernel_op_attributes(node.type)[0]).shape
        output_channel_axis, input_channel_axis = fw_info.kernel_channels_mapping.get(node.type)

        if node.type is Conv2D or node.type is Conv2DTranspose:
            # (C_out * W_out * H_out) * C_in * (W_kernel * H_kernel)
            return np.prod([x for x in output_shape if x is not None]) * \
                   input_shape[input_channel_axis] * \
                   (kernel_shape[0] * kernel_shape[1])
        elif node.type is DepthwiseConv2D:
            # Depth * (W_out * H_out) * C_in * (W_kernel * H_kernel)
            return node.framework_attr.get(DEPTH_MULTIPLIER) * \
                   np.prod([x for x in output_shape if x is not None]) / output_shape[output_channel_axis] * \
                   input_shape[input_channel_axis] * \
                   (kernel_shape[0] * kernel_shape[1])
        elif node.type is Dense:
            # IN * OUT
            return kernel_shape[0] * kernel_shape[1]
        else:
            return 0

    def apply_second_moment_correction(self,
                                       quantized_model: Any,
                                       core_config: CoreConfig,
                                       representative_data_gen: Callable,
                                       graph: common.Graph):
        """
        Apply second moment statistics correction to graph.

        Args:
            quantized_model: Framework's model to apply second moment correction on.
            core_config: QuantizationConfig of how the model should be quantized.
            representative_data_gen: Dataset to use for retrieving images for the models inputs.
            graph: Graph to update the parameters after the second moment correction.

        Returns:
            A Graph after second moment correction.
        """
        graph_after_second_moment_correction = keras_apply_second_moment_correction(quantized_model, core_config,
                                                                                    representative_data_gen, graph)
        return graph_after_second_moment_correction
