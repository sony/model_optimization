from typing import List, Any, Tuple, Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from model_compression_toolkit import QuantizationConfig, FrameworkInfo, common, GradientPTQConfig, \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.common import Graph, BaseNode
from model_compression_toolkit.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.common.user_info import UserInformation
from model_compression_toolkit.keras.back2framework.model_builder import model_builder
from model_compression_toolkit.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.keras.gradient_ptq.training_wrapper import gptq_training_wrapper
from model_compression_toolkit.keras.graph_substitutions.substitutions.activation_decomposition import \
    ActivationDecomposition
from model_compression_toolkit.keras.graph_substitutions.substitutions.batchnorm_folding import \
    BatchNormalizationFolding
from model_compression_toolkit.keras.graph_substitutions.substitutions.input_scaling import InputScaling, \
    InputScalingWithPad
from model_compression_toolkit.keras.graph_substitutions.substitutions.mark_activation import MarkActivation
from model_compression_toolkit.keras.graph_substitutions.substitutions.relu_bound_correction import \
    ReLUBoundCorrection
from model_compression_toolkit.keras.graph_substitutions.substitutions.remove_relu_upper_bound import \
    RemoveReLUUpperBound
from model_compression_toolkit.keras.graph_substitutions.substitutions.scale_equalization import \
    ScaleEqualization, ScaleEqualizationWithPad, ScaleEqualizationMidActivation, ScaleEqualizationMidActivationWithPad
from model_compression_toolkit.keras.graph_substitutions.substitutions.separableconv_decomposition import \
    SeparableConvDecomposition
from model_compression_toolkit.keras.graph_substitutions.substitutions.shift_negative_activation import \
    apply_shift_negative_correction
from model_compression_toolkit.keras.mixed_precision.sensitivity_evaluation import get_sensitivity_evaluation
from model_compression_toolkit.keras.reader.reader import model_reader
from model_compression_toolkit.keras.tensor_marking import get_node_stats_collector
import model_compression_toolkit.keras.constants as keras_constants


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

    def model_reader(self, model: Model) -> Graph:
        """
        Convert a framework's model into a graph.
        Args:
            model: Framework's model.

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
        return tensor.numpy()

    def model_builder(self,
                      graph: Graph,
                      mode: ModelBuilderMode,
                      append2output: List[Any] = None,
                      fw_info: FrameworkInfo = DEFAULT_KERAS_INFO) -> Tuple[Model, UserInformation]:
        """
        Build a Keras model from a graph.
        The mode determines how the model should be build. append2output is a list of Nodes
        to set as the model outputs.

        Args:
            graph: Graph to build the model from it.
            mode: Mode for how to build the model.
            append2output: List of Nodes to set as the model's outputs.
            fw_info: FrameworkInfo object with information about the specific framework's model

        Returns:
            A tuple of the Keras model that was built and an UserInformation object.
        """
        return model_builder(graph,
                             mode,
                             append2output,
                             fw_info)

    def shift_negative_correction(self,
                                  graph: Graph,
                                  qc: QuantizationConfig,
                                  fw_info: FrameworkInfo) -> Graph:
        """
        Apply shift negative correction (SNC) on a graph.

        Args:
            graph: Graph to apply SNC on.
            qc: Quantization configuration.
            fw_info: FrameworkInfo object with information about the specific framework's model.

        Returns:
            Graph after SNC.
        """
        return apply_shift_negative_correction(graph,
                                               qc,
                                               fw_info)

    def attach_sc_to_node(self, node: BaseNode,
                          fw_info: FrameworkInfo) -> common.statistics_collector.BaseStatsContainer:
        """
        Return a statistics collector that should be attached to a node's output
        during statistics collection.

        Args:
            node: Node to return its collector.
            fw_info: FrameworkInfo object with information about the specific framework's model

        Returns:
            Statistics collector for the node.
        """
        return get_node_stats_collector(node,
                                        fw_info)

    def get_substitutions_marking(self) -> List[common.BaseSubstitution]:
        """

        Returns: A list of the framework substitutions used for marking
        points we fuse.

        """
        return [MarkActivation()]

    def get_substitutions_pre_statistics_collection(self) -> List[common.BaseSubstitution]:
        """

        Returns: A list of the framework substitutions used before we build a quantized model.

        """
        return [SeparableConvDecomposition(),
                ActivationDecomposition(),
                BatchNormalizationFolding()]

    def get_substitutions_post_statistics_collection(self, quant_config: QuantizationConfig) -> List[
        common.BaseSubstitution]:
        """
        Return a list of the framework substitutions used after we collect statistics.

        Args:
            quant_config: QuantizationConfig to determine which substitutions to return.

        Returns:
            A list of the framework substitutions used after we collect statistics.
        """
        substitutions_list = []
        if quant_config.input_scaling:
            substitutions_list.append(InputScaling())
            substitutions_list.append(InputScalingWithPad())

        if quant_config.relu_unbound_correction:
            substitutions_list.append(ReLUBoundCorrection())
        return substitutions_list

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

    def get_substitutions_pre_build(self) -> List[common.BaseSubstitution]:
        """

        Returns: A list of the framework substitutions used before we build a quantized model.

        """

        return [RemoveReLUUpperBound()]

    def gptq_training(self,
                      graph: Graph,
                      representative_data_gen: Callable,
                      gptq_config: GradientPTQConfig,
                      fw_info: FrameworkInfo) -> Graph:
        """
        Update a graph using GPTQ after minimizing the loss between the float model's output
        and the quantized model's outputs.

        Args:
            graph: Graph to fine-tune.
            representative_data_gen: Dataset to use for inputs of the models.
            gptq_config: GradientPTQConfig with configuration for the fine-tuning process.
            fw_info: FrameworkInfo object with information about the specific framework's model.

        Returns:
            Updated graph after GPTQ.
        """

        return gptq_training_wrapper(graph,
                                     representative_data_gen,
                                     gptq_config,
                                     fw_info)


    def get_sensitivity_evaluation_fn(self,
                                      graph: Graph,
                                      quant_config: MixedPrecisionQuantizationConfig,
                                      metrics_weights: np.ndarray,
                                      representative_data_gen: Callable,
                                      fw_info: FrameworkInfo) -> Callable:
        """
        Create and return a function to compute a sensitivity metric for a mixed-precision
        configuration (comparing to the float Keras model).

        Args:
            graph: Graph to build it's float and mixed-precision Keras models.
            quant_config: QuantizationConfig of how the model should be quantized.
            metrics_weights: Array of weights to weight the sensitivity among different layers.
            representative_data_gen: Dataset to use for retrieving images for the models inputs.
            fw_info: FrameworkInfo object with information about the specific framework's model.

        Returns:
            A function that computes the metric.
        """

        return get_sensitivity_evaluation(graph,
                                          quant_config,
                                          metrics_weights,
                                          representative_data_gen,
                                          fw_info)
