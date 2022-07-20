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
import copy
from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, List, Any
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig
from model_compression_toolkit.core.common import Graph, Logger
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.gptq.common.gptq_graph import get_compare_points
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode


class GPTQTrainer(ABC):
    """
    Abstract GPTQ training class for fine-tuning a quantized model
    """

    def __init__(self,
                 graph_float: Graph,
                 graph_quant: Graph,
                 gptq_config: GradientPTQConfig,
                 fw_impl: FrameworkImplementation,
                 fw_info: FrameworkInfo):
        """
        Build two models from a graph: A teacher network (float model) and a student network (quantized model).
        Use the dataset generator to pass images through the teacher and student networks to get intermediate
        layers outputs. Use the outputs to compute the observed loss and to back-propagate the error
        in the student network, to minimize it in the next similar steps.
        All parameters (such as number of iterations, optimizer, etc.) are in GradientPTQConfig.
        Args:
            graph_float: Graph to build a float networks from.
            graph_quant: Graph to build a quantized networks from.
            gptq_config: GradientPTQConfig with parameters about the tuning process.
            fw_impl: Framework implementation
            fw_info: Framework information
        """
        self.graph_float = copy.deepcopy(graph_float)
        self.graph_quant = copy.deepcopy(graph_quant)
        self.gptq_config = gptq_config
        self.fw_impl = fw_impl
        self.fw_info = fw_info

        # ----------------------------------------------
        # Build two models and create compare nodes
        # ----------------------------------------------
        self.compare_points, _, self.compare_points_mean, self.compare_points_std = get_compare_points(self.graph_float)

        self.float_model, self.float_user_info = fw_impl.model_builder(self.graph_float,
                                                                       mode=ModelBuilderMode.FLOAT,
                                                                       append2output=self.compare_points,
                                                                       fw_info=self.fw_info)

        self.fxp_model, self.gptq_user_info = self.build_gptq_model()

    def get_optimizer_with_param(self,
                                 flattened_trainable_weights: List[Any],
                                 flattened_bias_weights: List[Any],
                                 trainable_quantization_parameters: List[Any],
                                 temperature_weights: List[Any]) -> List[Any]:
        """
        Create Optimizers with their trainable parameters
        Args:
            flattened_trainable_weights: list of trainable weights parameters (flattened)
            flattened_bias_weights: list of trainable bias parameters (flattened)
            trainable_quantization_parameters: list of trainable quantization parameters
            temperature_weights: list of temperature weights variables
        Returns:
            List of Optimizer objects with parameters
        """

        w2train = [*flattened_trainable_weights]
        if self.gptq_config.is_gumbel:
            if self.gptq_config.quantizer_config.temperature_learning:
                w2train.extend(temperature_weights)

        optimizer_with_param = [(self.gptq_config.optimizer, w2train)]
        if self.gptq_config.train_bias or self.gptq_config.quantization_parameters_learning:
            w2train_res = []
            if self.gptq_config.train_bias:
                if self.gptq_config.optimizer_bias is not None:
                    optimizer_with_param.append((self.gptq_config.optimizer_bias, flattened_bias_weights))
                else:
                    w2train_res.extend(flattened_bias_weights)
                    if self.gptq_config.optimizer_rest is None:
                        Logger.error(
                            "To enable bias micro training an additional optimizer is required, please define the optimizer_rest")
            if self.gptq_config.quantization_parameters_learning:
                if self.gptq_config.optimizer_quantization_parameter is not None:  # Ability to override optimizer
                    optimizer_with_param.append((self.gptq_config.optimizer_quantization_parameter,
                                                 trainable_quantization_parameters))
                else:
                    w2train_res.extend(trainable_quantization_parameters)
                if self.gptq_config.optimizer_rest is None:
                    Logger.error(
                        "To enable bias micro training an additional optimizer is required, please define the optimizer_rest")
            optimizer_with_param.append((self.gptq_config.optimizer_rest, w2train_res))

        return optimizer_with_param


    def compute_jacobian_based_weights(self,
                                       representative_data_gen: Callable) -> np.ndarray:
        """
        Computes the jacobian-based weights using the framework's model_grad method per batch of images.

        Args:
            representative_data_gen: Dataset used for inference to compute the jacobian-based weights.

        Returns: A vector of weights, one for each compare point,
        to be used for the loss metric weighted average computation when running GPTQ training.
        """
        if self.gptq_config.use_jac_based_weights:
            images = self._generate_images_batch(representative_data_gen, self.gptq_config.num_samples_for_loss)
            points_apprx_jacobians_weights = []
            for i in range(1, images.shape[0] + 1):
                # Note that in GPTQ loss weights computation we assume that there aren't replacement output nodes,
                # therefore, output_list is just the graph outputs, and we don't need the tuning factor for
                # defining the output weights (since the output layer is not a compare point).
                image_ip_gradients = self.fw_impl.model_grad(self.graph_float,
                                                             {inode: self.fw_impl.to_tensor(images[i - 1:i]) for inode in
                                                              self.graph_float.get_inputs()},
                                                             self.compare_points,
                                                             output_list=[n.node for n in
                                                                          self.graph_float.get_outputs()],
                                                             all_outputs_indices=[],
                                                             alpha=0,
                                                             norm_weights=self.gptq_config.norm_weights)
                points_apprx_jacobians_weights.append(image_ip_gradients)
            return np.mean(points_apprx_jacobians_weights, axis=0)
        else:
            num_nodes = len(self.compare_points)
            return np.asarray([1 / num_nodes for _ in range(num_nodes)])

    @staticmethod
    def _generate_images_batch(representative_data_gen: Callable, num_samples_for_loss: int) -> np.ndarray:
        """
        Construct batches of image samples for inference.

        Args:
            representative_data_gen: A callable method to retrieve images from Dataset.
            num_samples_for_loss: Num of total images for evaluation.

        Returns: A tensor of images batches
        """
        # First, select images to use for all measurements.
        samples_count = 0  # Number of images we used so far to compute the distance matrix.
        images = []
        while samples_count < num_samples_for_loss:
            # Get a batch of images to infer in both models.
            inference_batch_input = representative_data_gen()
            num_images = inference_batch_input[0].shape[0]

            # If we sampled more images than we should,
            # we take only a subset of these images and use only them.
            if num_images > num_samples_for_loss - samples_count:
                inference_batch_input = [x[:num_samples_for_loss - samples_count] for x in inference_batch_input]
                assert num_samples_for_loss - samples_count == inference_batch_input[0].shape[0]
                num_images = num_samples_for_loss - samples_count

            images.append(inference_batch_input[0])
            samples_count += num_images
        return np.concatenate(images, axis=0)


    @abstractmethod
    def build_gptq_model(self):
        """
        Build the GPTQ model with QuantizationWrappers
        Returns:
            Quantized graph for GPTQ fine-tuning, GPTQ graph user info
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s GPTQ model builder method.')

    @abstractmethod
    def train(self, representative_data_gen: Callable):
        """
        Train the quantized model using GPTQ training process
        Args:
            representative_data_gen: Dataset to use for inputs of the models.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s train method.')

    @abstractmethod
    def update_graph(self) -> Graph:
        """
        Update a graph using GPTQ after minimizing the loss between the float model's output
        and the quantized model's outputs.
        Returns:
            Updated graph after GPTQ.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s update_graph method.')


def gptq_training(graph_float: Graph,
                  graph_quant: Graph,
                  gptq_config: GradientPTQConfig,
                  representative_data_gen: Callable,
                  fw_impl: FrameworkImplementation,
                  fw_info: FrameworkInfo) -> Graph:
    """
    GPTQ training process using knowledge distillation with a teacher network (float model) and a student network (quantized model).
    Args:
        graph_float: Graph to build a float networks from.
        graph_quant: Graph to build a quantized networks from.
        gptq_config: GradientPTQConfig with parameters about the tuning process.
        representative_data_gen: Dataset to use for inputs of the models.
        fw_impl: Framework implementation
        fw_info: Framework information

    Returns:
        Quantized graph for export

    """
    # Get GPTQ object and initialize it
    gptq_trainer_obj = fw_impl.get_gptq_trainer_obj()

    gptq_trainer = gptq_trainer_obj(graph_float,
                                    graph_quant,
                                    gptq_config,
                                    fw_impl,
                                    fw_info,
                                    representative_data_gen)

    # Training process
    gptq_trainer.train(representative_data_gen)

    # Update graph
    graph_quant = gptq_trainer.update_graph()

    return graph_quant
