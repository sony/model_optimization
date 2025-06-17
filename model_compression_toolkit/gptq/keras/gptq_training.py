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
from typing import Callable, List, Tuple, Union, Generator

import tensorflow as tf
from keras import Model
from packaging import version
from tensorflow.keras.layers import Layer
from tqdm import tqdm

from model_compression_toolkit.core.common.hessian import HessianInfoService, HessianScoresGranularity
# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.keras.back2framework.keras_model_builder import KerasModelBuilder
from model_compression_toolkit.core.keras.data_util import data_gen_to_dataloader, \
    FixedSampleInfoDataset, FixedTFDataset, create_tf_dataloader, TFDatasetFromGenerator, \
    IterableSampleWithConstInfoDataset
from model_compression_toolkit.gptq.common.gptq_graph import get_kernel_attribute_name_for_gptq
from model_compression_toolkit.gptq.common.gradual_activation_quantization import \
    get_gradual_activation_quantizer_wrapper_factory
from model_compression_toolkit.gptq.common.regularization_factory import get_regularization
from model_compression_toolkit.gptq.keras.quantizer.quantization_builder import quantization_builder
from model_compression_toolkit.logger import Logger
from mct_quantizers import KerasActivationQuantizationHolder
from model_compression_toolkit.trainable_infrastructure.common.util import get_total_grad_steps
from model_compression_toolkit.trainable_infrastructure.keras.annealing_schedulers import KerasLinearAnnealingScheduler

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.engine.base_layer import TensorFlowOpLayer
else:
    from keras.engine.base_layer import TensorFlowOpLayer

from model_compression_toolkit.trainable_infrastructure import KerasTrainableQuantizationWrapper
from model_compression_toolkit.core import common
from model_compression_toolkit.gptq.common.gptq_training import GPTQTrainer
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.gptq.keras.graph_info import get_weights_for_loss, get_gptq_trainable_parameters
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
import numpy as np
import copy
from model_compression_toolkit.core.keras.constants import BIAS, USE_BIAS
from model_compression_toolkit.gptq.keras.quantizer.soft_rounding.soft_quantizer_reg import SoftQuantizerRegularization

class KerasGPTQTrainer(GPTQTrainer):
    """
    Keras GPTQ training class for fine-tuning a quantized model
    """

    def __init__(self,
                 graph_float: Graph,
                 graph_quant: Graph,
                 gptq_config: GradientPTQConfig,
                 fw_impl: FrameworkImplementation,
                 representative_data_gen: Callable,
                 hessian_info_service: HessianInfoService = None):
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
            fw_impl: FrameworkImplementation object with a specific framework methods implementation.
            representative_data_gen: Dataset to use for inputs of the models.
            hessian_info_service: HessianScoresService for fetching and computing Hessian's approximation scores.

        """

        self.fw_soft_quantizer_regularization = SoftQuantizerRegularization
        self.fw_linear_annealing_scheduler = KerasLinearAnnealingScheduler
        self.fw_get_gptq_trainable_parameters_fn = get_gptq_trainable_parameters
        self.fw_get_weights_for_loss_fn = get_weights_for_loss

        super().__init__(graph_float,
                         graph_quant,
                         gptq_config,
                         fw_impl,
                         representative_data_gen_fn=representative_data_gen,
                         hessian_info_service=hessian_info_service)


    def _prepare_train_dataloader_sla(self, data_gen_fn: Callable[[], Generator]) -> tf.data.Dataset:
        """
        Computes Sample-Layer Attention score and builds a train dataloader in TensorFlow.

        Args:
            data_gen_fn: function for representative dataset generation.

        Returns:
            TensorFlow dataset yielding three outputs - samples, weights for the distillation loss,
            and weights for regularization.
        """
        # Create a fixed dataset
        fixed_dataset = FixedTFDataset(data_gen_fn)
        orig_batch_size = fixed_dataset.orig_batch_size

        # Prepare a separate loader for computing hessians over the whole dataset
        hess_data_loader = create_tf_dataloader(
            fixed_dataset,
            batch_size=self.gptq_config.hessian_weights_config.hessian_batch_size,
            shuffle=False
        )

        # Prepare request for Hessian computation
        request = self._build_hessian_request(
            granularity=HessianScoresGranularity.PER_OUTPUT_CHANNEL,
            data_loader=hess_data_loader,
            n_samples=None
        )
        layers_hessians = self.hessian_service.fetch_hessian(request, force_compute=True)

        # Compute SLA score defined as max over elements
        layers_hessians = {
            layer: tf.convert_to_tensor(tf.reduce_max(hess, axis=tuple(range(1, len(hess.shape))))) for layer, hess in layers_hessians.items()
        }

        # Stack hessians for comparison points
        hessians_tensor = tf.stack([layers_hessians[layer.name] for layer in self.compare_points])
        assert hessians_tensor.shape[0] == len(self.compare_points)
        loss_weights = list(hessians_tensor.numpy())  # Convert to a list for compatibility

        # Prepare final dataset with samples and loss weights
        sla_train_dataset = FixedSampleInfoDataset(fixed_dataset.samples, loss_weights)

        # Calculate regularization weights as mean across samples
        reg_weights = tf.reduce_mean(hessians_tensor, axis=1)

        # Define a collate function to add regularization weights to each batch
        def collate_fn(samples_with_loss_weights):
            return *samples_with_loss_weights, reg_weights

        # Create final dataset using the new dataloader with collate_fn
        final_dataset = create_tf_dataloader(
            sla_train_dataset,
            batch_size=orig_batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        return final_dataset

    def _prepare_train_dataloader_for_non_sla(self,
                                              data_gen_fn: Callable[[], Generator]) -> tf.data.Dataset:
        """
        Prepares a train dataloader for non-SLA tasks.

        Args:
            data_gen_fn: Factory for representative dataset generator.

        Returns:
            A `tf.data.Dataset` yielding samples with loss weights and regularization weights.
        """
        # Step 1: Create a dataset from the generator
        dataset = TFDatasetFromGenerator(data_gen_fn)
        num_nodes = len(self.compare_points)

        # Step 2: Compute loss weights
        if self.gptq_config.hessian_weights_config:
            hessian_dataset = create_tf_dataloader(dataset, batch_size=self.gptq_config.hessian_weights_config.hessian_batch_size)
            hessian_weights = self.compute_hessian_based_weights(hessian_dataset)
            loss_weights = tf.convert_to_tensor(hessian_weights, dtype=tf.float32)
        else:
            loss_weights = tf.ones(num_nodes, dtype=tf.float32) / num_nodes

        # Step 3: Create a dataset with samples and loss weights
        augmented_dataset = IterableSampleWithConstInfoDataset(dataset.tf_dataset, loss_weights)

        # Step 4: Add constant regularization weights
        reg_weights = tf.ones(num_nodes, dtype=tf.float32)

        def collate_fn(batch):
            samples, loss_weights = batch
            return samples, loss_weights, reg_weights

        # Step 5: Create a tf.data.Dataset with collate_fn
        train_dataloader = create_tf_dataloader(augmented_dataset,
                                                batch_size=dataset.orig_batch_size,
                                                collate_fn=collate_fn)

        return train_dataloader

    def _is_gptq_weights_trainable(self,
                                   node: common.BaseNode) -> bool:
        """
        A function for deciding if a layer should be fine-tuned during GPTQ.

        Args:
            node (BaseNode): Node for quantization decision

        Returns:
            A boolean whether the layer is to be wrapped with a QuantizeWrapper
        """
        return node.kernel_attr is not None and node.is_weights_quantization_enabled(node.kernel_attr)

    def gptq_wrapper(self,
                     n: common.BaseNode,
                     layer: Layer) -> Union[KerasTrainableQuantizationWrapper, Layer]:
        """
        A function which takes a computational graph node and a keras layer and perform the quantization wrapping.

        Args:
            n: A node of mct graph.
            layer: A keras layer

        Returns: Wrapped layer if the layer should be wrap, otherwise returns the layer as is.

        """
        if self._is_gptq_weights_trainable(n):
            # If we are here, then the node has a kernel attribute to quantize and training during GPTQ
            weights_quantizers, _ = quantization_builder(n,
                                                         self.gptq_config,  # TODO: split quantizers building into two functions: for weights and activations
                                                         n.kernel_attr)
            if len(weights_quantizers) > 0:
                return KerasTrainableQuantizationWrapper(layer,
                                                         weights_quantizers=weights_quantizers)

        # TODO: need to check if in this case, if there are other weights attributes that are not trainable but are
        #  quantized, do we need to wrap them as well?
        return layer

    def get_activation_quantizer_holder(self, n: common.BaseNode) -> Callable:
        """
        Retrieve a KerasActivationQuantizationHolder layer to use for activation quantization for a node.
        If the layer is not supposed to be wrapped with activation quantizers - return None.

        Args:
            n: Node to get KerasActivationQuantizationHolder to attach in its output.

        Returns:
            A KerasActivationQuantizationHolder layer for the node activation quantization.
        """
        _, activation_quantizers = quantization_builder(n, self.gptq_config) # TODO: split quantizers building into two functions: for weights and activations

        # Holder by definition uses a single quantizer for the activation quantization
        # thus we make sure this is the only possible case.
        if len(activation_quantizers) != 1:
            Logger.critical(f"'KerasActivationQuantizationHolder' is designed to support a single quantizer, "
                            f"but {len(activation_quantizers)} quantizers were found for node '{n}'. "
                            f"Ensure only one quantizer is configured for each node's activation.")
        quantizer = self.gradual_act_quantizer_wrapper_factory(activation_quantizers[0])
        return KerasActivationQuantizationHolder(quantizer)

    def build_gptq_model(self) -> Tuple[Model, UserInformation]:
        """
        Build the GPTQ model with QuantizationWrappers

        Returns:
            Quantized graph for GPTQ fine-tuning, GPTQ graph user info
        """

        gptq_model, gptq_user_info = KerasModelBuilder(graph=self.graph_quant,
                                                       append2output=self.compare_points,
                                                       return_float_outputs=True,
                                                       wrapper=self.gptq_wrapper,
                                                       get_activation_quantizer_holder_fn=self.get_activation_quantizer_holder).build_model()

        return gptq_model, gptq_user_info

    def compute_gradients(self,
                          in_y_float: List[tf.Tensor],
                          input_data: List[np.ndarray],
                          in_optimizer_with_param: List,
                          training=True,
                          distill_loss_weights=None,
                          reg_weights=None) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Get outputs from both teacher and student networks. Compute the observed error,
        and use it to compute the gradients and applying them to the student weights.
        Args:
            in_y_float: A list of reference tensor from the floating point network.
            input_data: A list of Input tensors to pass through the networks.
            in_optimizer_with_param: A list of optimizer classes to update with the corresponding parameters.
            training: A boolean flag stating if the network is running in training mode.

        Returns:
            Loss and gradients.
        """
        param2grad = []
        for _, p in in_optimizer_with_param:
            param2grad.extend(p)

        with tf.GradientTape(persistent=True) as tape:
            y_fxp = self.fxp_model(input_data, training=training)  # running fxp model
            loss_value = self.gptq_config.loss(y_fxp,
                                               in_y_float,
                                               self.fxp_weights_list,
                                               self.flp_weights_list,
                                               self.compare_points_mean,
                                               self.compare_points_std,
                                               distill_loss_weights)

            reg_value = self.reg_func(self.fxp_model, self.gptq_config.regularization_factor, reg_weights)

            loss_value += reg_value

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, param2grad)
        res = []
        i = 0
        for _, p in in_optimizer_with_param:
            res.append(grads[i:(i + len(p))])
            i += len(p)
        return loss_value, res

    def train(self):
        """
        Train the quantized model using GPTQ training process in Keras framework
        """
        compute_gradients = self.compute_gradients

        # ----------------------------------------------
        # Training loop
        # ----------------------------------------------
        if self.has_params_to_train:
            self.micro_training_loop(compute_gradients,
                                     self.optimizer_with_param,
                                     self.gptq_config.n_epochs,
                                     True)

    @tf.function
    def nano_training_step(self,
                           input_data,
                           in_compute_gradients,
                           in_optimizer_with_param,
                           is_training,
                           distill_loss_weights,
                           reg_weights):
        """
        This function run part of the training step, wrapped by a tf.function for acceleration.
        Args:
            input_data: input data for the step.
            in_compute_gradients: A callable function that compute the gradients.
            in_optimizer_with_param: A list of optimizer classes to update with the corresponding parameters.
            is_training: A boolean flag stating if the network is running in training mode.

        Returns:
            loss value and gradients

        """

        # run float model
        y_float = self.float_model(input_data)
        # rung quantized model and calculate loss & gradients
        loss_value_step, grads = in_compute_gradients(y_float,
                                                      input_data,
                                                      in_optimizer_with_param,
                                                      training=is_training,
                                                      distill_loss_weights=distill_loss_weights,
                                                      reg_weights=reg_weights)
        return loss_value_step, grads

    def micro_training_loop(self,
                            in_compute_gradients: Callable,
                            in_optimizer_with_param: List[Tuple[tf.keras.optimizers.Optimizer, List[tf.Tensor]]],
                            n_epochs: int,
                            is_training: bool):
        """
        This function run a micro training loop on given set of parameters.
        Args:
            in_compute_gradients: A callable function that compute the gradients.
            in_optimizer_with_param: A list of optimizer classes to update with the corresponding parameters.
            n_epochs: Number of update iterations of representative dataset.
            is_training: A boolean flag stating if the network is running in training mode.

        Returns: None

        """
        with tqdm(range(n_epochs), "Running GPTQ optimization") as epochs_pbar:
            for _ in epochs_pbar:
                with tqdm(self.train_dataloader, position=1, leave=False) as data_pbar:
                    for data in data_pbar:

                        input_data, distill_loss_weights, reg_weight = data

                        input_data = [d * self.input_scale for d in input_data]

                        loss_value_step, grads = self.nano_training_step(input_data,
                                                                         in_compute_gradients,
                                                                         in_optimizer_with_param,
                                                                         is_training,
                                                                         distill_loss_weights,
                                                                         reg_weight)
                        # Run one step of gradient descent by updating
                        # the value of the variables to minimize the loss.
                        for i, (o, p) in enumerate(in_optimizer_with_param):
                            o.apply_gradients(zip(grads[i], p))
                        if self.gptq_config.log_function is not None:
                            self.gptq_config.log_function(loss_value_step, grads[0], in_optimizer_with_param[0][-1],
                                                          self.compare_points)
                        self.loss_list.append(loss_value_step.numpy())
                        Logger.debug(f'last loss value: {self.loss_list[-1]}')

    def update_graph(self):
        """
        Update a graph using GPTQ after minimizing the loss between the float model's output
        and the quantized model's outputs.
        Returns:
            Updated graph after GPTQ.
        """
        graph = copy.copy(self.graph_quant)

        for layer in self.fxp_model.layers:
            if isinstance(layer, KerasTrainableQuantizationWrapper):
                node = graph.find_node_by_name(layer.layer.name)
                if len(node) == 0 and isinstance(layer.layer, TensorFlowOpLayer):
                    node = graph.find_node_by_name('_'.join(layer.layer.name.split('_')[3:]))
                if len(node) != 1:
                    Logger.critical(f"Unable to update the GPTQ graph because the layer named '{layer.layer.name}' could not be found. "
                                    f"Verify that the layer names in the GPTQ model match those in the graph.")
                node = node[0]
                kernel_attribute = get_kernel_attribute_name_for_gptq(layer_type=node.type)
                # TODO: only kernel attributes are currently trained in GPTQ, so only the kernel weights need to be updated.
                #  To enable GPTQ for other attributes, this code needs to be modified.
                weights, weight_quant_config, activation_quant_config = \
                    layer.weights_quantizers[kernel_attribute].update_layer_quantization_params(layer)
                for weight_attr, weight in weights.items():
                    node.set_weights_by_keys(weight_attr, weight.numpy())
                for config_parameter_name, config_parameter_value in weight_quant_config.items():
                    node.final_weights_quantization_cfg.set_quant_config_attr(config_parameter_name,
                                                                              config_parameter_value,
                                                                              attr_name=kernel_attribute)
                for config_attr, config_value in activation_quant_config.items():
                    node.final_activation_quantization_cfg.set_quant_config_attr(config_attr, config_value)
                if self.gptq_config.train_bias:
                    use_bias = layer.layer.get_config().get(USE_BIAS)
                    if use_bias is not None and use_bias and layer.layer.bias is not None:
                        new_bias = layer.layer.bias.numpy()
                        node.set_weights_by_keys(BIAS, new_bias)

        return graph
