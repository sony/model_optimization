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
from functools import partial
from typing import List, Any, Tuple, Callable, Dict, Union, Generator

import numpy as np
import tensorflow as tf
from mct_quantizers import KerasQuantizationWrapper, KerasActivationQuantizationHolder
from tensorflow.keras.models import Model

from model_compression_toolkit.constants import HESSIAN_NUM_ITERATIONS
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.common.hessian import HessianScoresRequest, HessianMode, HessianInfoService
from model_compression_toolkit.core.keras.data_util import data_gen_to_dataloader
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.remove_identity import RemoveIdentity
from model_compression_toolkit.core.keras.hessian.activation_hessian_scores_calculator_keras import \
    ActivationHessianScoresCalculatorKeras
from model_compression_toolkit.core.keras.hessian.weights_hessian_scores_calculator_keras import WeightsHessianScoresCalculatorKeras
from model_compression_toolkit.core.keras.statistics_correction.keras_compute_activation_bias_correction_of_graph import \
    keras_compute_activation_bias_correction_of_graph
from model_compression_toolkit.exporter.model_wrapper.fw_agnostic.get_inferable_quantizers import \
    get_inferable_quantizers
from model_compression_toolkit.exporter.model_wrapper.keras.builder.node_to_quantizer import \
    get_weights_quantizer_for_node, get_activations_quantizer_for_node
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.mixed_precision.sensitivity_evaluation import SensitivityEvaluation
from model_compression_toolkit.core.common.mixed_precision.set_layer_to_bitwidth import set_layer_to_bitwidth
from model_compression_toolkit.core.common.similarity_analyzer import compute_kl_divergence, compute_cs, compute_mse
from model_compression_toolkit.core.keras.constants import ACTIVATION, SOFTMAX, SIGMOID, ARGMAX, LAYER_NAME, \
    COMBINED_NMS
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.batchnorm_reconstruction import \
    keras_batchnorm_reconstruction
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.virtual_activation_weights_composition import \
    VirtualActivationWeightsComposition
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.weights_activation_split import \
    WeightsActivationSplit
from model_compression_toolkit.core.keras.mixed_precision.configurable_activation_quantizer import \
    ConfigurableActivationQuantizer
from model_compression_toolkit.core.keras.mixed_precision.configurable_weights_quantizer import \
    ConfigurableWeightsQuantizer
from model_compression_toolkit.core.keras.statistics_correction.apply_second_moment_correction import \
    keras_apply_second_moment_correction
from packaging import version

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers import Dense, Activation, Conv2D, DepthwiseConv2D, Conv2DTranspose, Concatenate, Add
    from keras.src.layers.core import TFOpLambda
else:
    from keras.layers import Dense, Activation, Conv2D, DepthwiseConv2D, Conv2DTranspose, Concatenate, Add   # pragma: no cover
    from keras.layers.core import TFOpLambda   # pragma: no cover

from model_compression_toolkit.core import QuantizationConfig, FrameworkInfo, CoreConfig, MixedPrecisionQuantizationConfig
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.common.node_prior_info import NodePriorInfo
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.activation_decomposition import \
    ActivationDecomposition
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.matmul_substitution import \
    MatmulToDenseSubstitution
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.sigmoid_mul_to_swish import MulSigmoidToSwish
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.conv_funcs_to_layer import \
    Conv2dFuncToConv2dLayer, DwConv2dFuncToDwConv2dLayer
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.softmax_shift import \
    keras_softmax_shift
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.batchnorm_folding import \
    keras_batchnorm_folding, keras_batchnorm_forward_folding
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.batchnorm_refusing import \
    keras_batchnorm_refusing
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.linear_collapsing import \
    keras_linear_collapsing, keras_op2d_add_const_collapsing
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.residual_collapsing import \
    keras_residual_collapsing
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.input_scaling import InputScaling, \
    InputScalingWithPad
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.concat_threshold_update import ConcatThresholdUpdate
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.relu_bound_to_power_of_2 import \
    ReLUBoundToPowerOfTwo
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.multi_head_attention_decomposition import \
    MultiHeadAttentionDecomposition
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.scale_equalization import \
    ScaleEqualization, ScaleEqualizationWithPad, ScaleEqualizationMidActivation, ScaleEqualizationMidActivationWithPad
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.separableconv_decomposition import \
    SeparableConvDecomposition, DEPTH_MULTIPLIER
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.shift_negative_activation import \
    keras_apply_shift_negative_correction
from model_compression_toolkit.core.keras.graph_substitutions.substitutions.dwconv_to_conv import DwconvToConv
from model_compression_toolkit.core.keras.keras_node_prior_info import create_node_prior_info
from model_compression_toolkit.core.keras.reader.reader import model_reader
import model_compression_toolkit.core.keras.constants as keras_constants
from model_compression_toolkit.core.keras.tf_tensor_numpy import tf_tensor_to_numpy, to_tf_tensor
from model_compression_toolkit.core.keras.back2framework import get_keras_model_builder


class KerasImplementation(FrameworkImplementation):
    """
    A class with implemented methods to support optimizing Keras models.
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

    def is_tuple_of_tensors(self, obj: Any) -> bool:
        """
        Check if a given object if a tuple of tensors
        :param obj: Object to check its type
        :return: True if obj is a tuple of tensors, False otherwise
        """
        if not isinstance(obj, tuple):
            return False
        for item in obj:
            if not isinstance(item, tf.Tensor):
                return False
        return True

    def model_builder(self,
                      graph: Graph,
                      mode: ModelBuilderMode,
                      append2output: List[Any] = None,
                      fw_info: FrameworkInfo = DEFAULT_KERAS_INFO,
                      return_float_outputs: bool = False) -> Tuple:
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
            A tuple with the model and additional relevant supporting objects.
        """

        keras_model_builder = get_keras_model_builder(mode)
        return keras_model_builder(graph=graph,
                                   append2output=append2output,
                                   fw_info=fw_info,
                                   return_float_outputs=return_float_outputs).build_model()

    def run_model_inference(self,
                            model: Any,
                            input_list: List[Any],
                            requires_grad: bool = False) -> Tuple[tf.Tensor]:
        """
        Runs inference on the given Keras model with the provided inputs.

        This method executes the model on the given input data. If `requires_grad` is set to `False`,
        gradients will not be computed during inference by wrapping execution in a `tf.stop_gradient()` context.

        Args:
            model: The Keras model to execute.
            input_list: A list of inputs for the model.
            requires_grad: If False, prevents gradient computation (default: False).

        Returns:
            A tuple containing the model's output tensors.
        """
        # Prevent gradient computation if requires_grad is False
        if requires_grad:
            # Record operations for automatic differentiation
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
                g.watch(input_list)
                return model(input_list)
        else:
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

    def compute_activation_bias_correction(self,
                                           graph: Graph,
                                           quant_config: QuantizationConfig,
                                           fw_info: FrameworkInfo):
        """
        Compute activation bias correction on a graph.

        Args:
            graph: Graph to apply activation bias correction on.
            quant_config: QuantizationConfig of how the model should be quantized.
            fw_info: FrameworkInfo object with information about the specific framework's model.

        Returns:
            Graph after activation bias correction computing.
        """
        return keras_compute_activation_bias_correction_of_graph(graph=graph,
                                                                 quant_config=quant_config,
                                                                 fw_info=fw_info,
                                                                 fw_impl=self)

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
        return [MulSigmoidToSwish(),
                SeparableConvDecomposition(),
                MatmulToDenseSubstitution(),
                Conv2dFuncToConv2dLayer(),
                DwConv2dFuncToDwConv2dLayer(),
                MultiHeadAttentionDecomposition(),
                ActivationDecomposition(),
                DwconvToConv(),
                RemoveIdentity()]

    def get_substitutions_pre_statistics_collection(self, quant_config: QuantizationConfig) -> \
            List[common.BaseSubstitution]:
        """
        Return a list of the framework substitutions used before we collect statistics.

        Args:
            quant_config: QuantizationConfig to determine which substitutions to return.

        Returns:
            A list of the framework substitutions used before we collect statistics.

        """
        substitutions_list = [keras_batchnorm_folding(),
                              keras_batchnorm_forward_folding()]
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

    def get_op2d_add_const_collapsing_substitution(self) -> common.BaseSubstitution:
        """
        Returns: Op2d add-const collapsing substitution
        """
        return keras_op2d_add_const_collapsing()

    def get_substitutions_post_statistics_collection(self,
                                                     quant_config: QuantizationConfig) -> List[common.BaseSubstitution]:
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
        if quant_config.concat_threshold_update:
            substitutions_list.append(ConcatThresholdUpdate())
        return substitutions_list


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

    def get_sensitivity_evaluator(self,
                                  graph: Graph,
                                  quant_config: MixedPrecisionQuantizationConfig,
                                  representative_data_gen: Callable,
                                  fw_info: FrameworkInfo,
                                  disable_activation_for_metric: bool = False,
                                  hessian_info_service: HessianInfoService = None) -> SensitivityEvaluation:
        """
        Creates and returns an object which handles the computation of a sensitivity metric for a mixed-precision
        configuration (comparing to the float model).

        Args:
            graph: Graph to build its float and mixed-precision models.
            quant_config: QuantizationConfig of how the model should be quantized.
            representative_data_gen: Dataset to use for retrieving images for the models inputs.
            fw_info: FrameworkInfo object with information about the specific framework's model.
            disable_activation_for_metric: Whether to disable activation quantization when computing the MP metric.
            hessian_info_service: HessianScoresService to fetch scores based on a Hessian-approximation for the float model.

        Returns:
            A SensitivityEvaluation object.
        """

        return SensitivityEvaluation(graph=graph,
                                     quant_config=quant_config,
                                     representative_data_gen=representative_data_gen,
                                     fw_info=fw_info,
                                     fw_impl=self,
                                     set_layer_to_bitwidth=partial(set_layer_to_bitwidth,
                                                                   weights_quantizer_type=ConfigurableWeightsQuantizer,
                                                                   activation_quantizer_type=ConfigurableActivationQuantizer,
                                                                   weights_quant_layer_type=KerasQuantizationWrapper,
                                                                   activation_quant_layer_type=KerasActivationQuantizationHolder),
                                     disable_activation_for_metric=disable_activation_for_metric,
                                     hessian_info_service=hessian_info_service)

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
        if self.is_softmax(node) or self.is_sigmoid(node):
            return True

        return any([node.is_match_type(_type) for _type in [Conv2D, DepthwiseConv2D, Conv2DTranspose, Dense,
                                                            Concatenate, tf.concat, Add, tf.add, tf.stack]])

    def get_mp_node_distance_fn(self, n: BaseNode,
                                compute_distance_fn: Callable = None,
                                norm_mse: bool = False) -> Tuple[Callable, int]:
        """
        A mapping between layers' types and a distance function for computing the distance between
        two tensors in mixed precision (for loss computation purposes). Returns a specific function if node of specific types is
        given, or a default (normalized MSE) function otherwise.

        Args:
            n: Node to choose distance function for.
            compute_distance_fn: An optional distance function to use globally for all nodes.
            norm_mse: whether to normalize mse distance function.

        Returns: A distance function between two tensors and a axis on which the distance is computed (if exists).
        """

        axis = n.op_call_kwargs.get(keras_constants.AXIS) if isinstance(n, FunctionalNode) else n.framework_attr.get(keras_constants.AXIS)

        if compute_distance_fn is not None:
            return compute_distance_fn, axis

        # TODO should we really return mse if axis is None? Error? Fill default?
        if self.is_softmax(n) and axis is not None:
            return compute_kl_divergence, axis

        if self.is_sigmoid(n) or n.layer_class == Dense:
            return compute_cs, axis

        return partial(compute_mse, norm=norm_mse), axis

    @staticmethod
    def is_sigmoid(node: BaseNode):
        cls = node.layer_class
        return ((cls == Activation and node.framework_attr[ACTIVATION] == SIGMOID) or
                cls == tf.nn.sigmoid or
                cls == TFOpLambda and SIGMOID in node.framework_attr[keras_constants.FUNCTION])

    @staticmethod
    def is_softmax(node: BaseNode):
        cls = node.layer_class
        return ((cls == Activation and node.framework_attr[ACTIVATION] == SOFTMAX) or
                cls in [tf.nn.softmax, tf.keras.layers.Softmax] or
                cls == TFOpLambda and SOFTMAX in node.framework_attr[keras_constants.FUNCTION])

    def get_hessian_scores_calculator(self,
                                      graph: Graph,
                                      input_images: List[Any],
                                      hessian_scores_request: HessianScoresRequest,
                                      num_iterations_for_approximation: int = HESSIAN_NUM_ITERATIONS):
        """
        Get Keras Hessian-approximation scores calculator based on the request.
        Args:
            input_images: Images to use for computation.
            graph: Float graph to compute the approximation of its different nodes.
            hessian_scores_request: HessianScoresRequest to search for the desired calculator.
            num_iterations_for_approximation: Number of iterations to use when approximating the Hessian scores.

        Returns: HessianScoresCalculatorKeras to use for the Hessian-approximation scores computation for this request.

        """
        if hessian_scores_request.mode == HessianMode.ACTIVATION:
            return ActivationHessianScoresCalculatorKeras(graph=graph,
                                                          hessian_scores_request=hessian_scores_request,
                                                          input_images=input_images,
                                                          fw_impl=self,
                                                          num_iterations_for_approximation=num_iterations_for_approximation)
        elif hessian_scores_request.mode == HessianMode.WEIGHTS:
            return WeightsHessianScoresCalculatorKeras(graph=graph,
                                                       hessian_scores_request=hessian_scores_request,
                                                       input_images=input_images,
                                                       fw_impl=self,
                                                       num_iterations_for_approximation=num_iterations_for_approximation)
        else:
            Logger.critical(f"Unsupported Hessian mode for Keras: {hessian_scores_request.mode}.")   # pragma: no cover

    def is_output_node_compatible_for_hessian_score_computation(self,
                                                                node: BaseNode) -> Any:
        """
        Checks and returns whether the given node is compatible as output for Hessian-based information computation.

        Args:
            node: A BaseNode object.

        Returns: Whether the node is compatible as output for Hessian-based information computation.

        """

        if node.layer_class == TFOpLambda:
            node_attr = getattr(node, 'framework_attr', None)
            if node_attr is not None and (ARGMAX in node_attr[LAYER_NAME]
                                          or COMBINED_NMS in node_attr[LAYER_NAME]):
                return False
        elif node.layer_class in [tf.math.argmax]:
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
        kernels = fw_info.get_kernel_op_attributes(node.type)
        if not kernels or kernels[0] is None:
            return 0

        assert len(kernels) == 1
        kernel_shape = node.get_weights_by_keys(kernels[0]).shape

        if node.is_match_type(Conv2D) or node.is_match_type(Conv2DTranspose) or node.is_match_type(DepthwiseConv2D):
            h, w = node.get_output_shapes_list()[0][-3:-1]
            return np.prod(kernel_shape) * h * w

        if node.is_match_type(Dense):
            # IN * OUT * (all previous dims[:-1])
            _, input_channel_axis = fw_info.kernel_channels_mapping.get(node.type)
            return node.get_total_output_params() * kernel_shape[input_channel_axis]

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

    def sensitivity_eval_inference(self,
                                   model: Model,
                                   inputs: Any):
        """
        Calls for a Keras model inference for a specific framework during mixed precision sensitivity evaluation.

        Args:
            model: A Keras model to run inference for.
            inputs: Input tensors to run inference on.

        Returns:
            The output of the model inference on the given input.
        """

        return model(inputs)

    def get_inferable_quantizers(self, node: BaseNode):
        """
        Returns sets of Keras compatible weights and activation quantizers for the given node.

        Args:
           node: Node to get quantizers for.

        Returns:
            weight_quantizers: A dictionary between a weight's name to its quantizer.
            activation_quantizers: A list of activations quantization, one for each layer output.
        """

        def _weight_name(w: Union[str, int]) -> Union[str, int]:
            """
            Extracts the weight name from the full TensorFlow variable name.

            For example, returns 'kernel' for 'dense_2/kernel:0'.

            Args:
              w: TensorFlow variable name.

            Returns:
              Extracted weight name.
            """

            return w.split(':')[0].split('/')[-1] if isinstance(w, str) else w

        attribute_names = [_weight_name(wn) for wn in node.get_node_weights_attributes()
                           if node.is_weights_quantization_enabled(wn)]

        return get_inferable_quantizers(node,
                                        get_weights_quantizer_for_node,
                                        get_activations_quantizer_for_node,
                                        attribute_names)

    @staticmethod
    def convert_data_gen_to_dataloader(data_gen_fn: Callable[[], Generator], batch_size: int):
        """ Create DataLoader based on samples yielded by data_gen. """
        return data_gen_to_dataloader(data_gen_fn, batch_size=batch_size)
