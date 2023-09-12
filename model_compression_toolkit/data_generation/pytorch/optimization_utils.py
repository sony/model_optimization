# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import List, Callable, Type, Any, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize

from model_compression_toolkit.core.pytorch.utils import get_working_device
from model_compression_toolkit.data_generation.common.enums import ImageGranularity
from model_compression_toolkit.data_generation.common.image_pipeline import BaseImagePipeline
from model_compression_toolkit.data_generation.common.optimization_utils import BatchStatsHolder, AllImagesStatsHolder, \
    BatchOptimizationHolder, ImagesOptimizationHandler
from model_compression_toolkit.data_generation.common.constants import IMAGE_INPUT
from model_compression_toolkit.data_generation.pytorch.constants import BATCH_AXIS, H_AXIS, W_AXIS
from model_compression_toolkit.data_generation.pytorch.model_info_exctractors import ActivationExtractor


class PytorchImagesOptimizationHandler(ImagesOptimizationHandler):
    """
    An extension of ImagesOptimizationHandler specifically for PyTorch models.
    Handles the optimization process for generating images. Manages the order for which
    the image batches are optimized per iteration.

    Methods for a specific batch (specified by batch index):
    - Clear gradients.
    - Compute the batch-norm loss.
    - Perform a single optimization step.
    - Updates batch statistics.

    """
    def __init__(self,
                 model: Module,
                 data_gen_batch_size: int,
                 init_dataset: DataLoader,
                 optimizer: Optimizer,
                 image_pipeline: BaseImagePipeline,
                 activation_extractor: ActivationExtractor,
                 image_granularity: ImageGranularity,
                 scheduler_step_fn: Callable,
                 scheduler: Any,
                 initial_lr: float,
                 normalization_mean: List[float],
                 normalization_std: List[float],
                 clip_images: bool,
                 reflection: bool,
                 eps: float = 1e-6):
        """
        Constructor for the PytorchImagesOptimizationHandler class.

        Args:
            model (Module): The PyTorch model.
            data_gen_batch_size (int): Batch size for generating data.
            init_dataset (DataLoader): The initial dataset used for image generation.
            optimizer (Optimizer): The optimizer for updating the model parameters.
            image_pipeline (BaseImagePipeline): The image pipeline for processing images.
            activation_extractor (ActivationExtractor): Extractor for layer activations.
            image_granularity (ImageGranularity): The granularity of the images.
            scheduler_step_fn (Callable): The function to perform a scheduler step.
            scheduler (Any): The scheduler responsible for adjusting the learning rate of the optimizer over time.
            initial_lr (float): The initial learning rate used by the optimizer.
            normalization_mean (List[float]): The mean values for image normalization.
            normalization_std (List[float]): The standard deviation values for image normalization.
            clip_images (bool): Whether to clip the images during optimization.
            reflection (bool): Whether to use reflection during image clipping.
            eps (float): A small value added for numerical stability.
        """
        super(PytorchImagesOptimizationHandler, self).__init__(model=model,
                                                                  data_gen_batch_size=data_gen_batch_size,
                                                                  init_dataset=init_dataset,
                                                                  optimizer=optimizer,
                                                                  image_pipeline=image_pipeline,
                                                                  activation_extractor=activation_extractor,
                                                                  image_granularity=image_granularity,
                                                                  scheduler_step_fn=scheduler_step_fn,
                                                                  scheduler=scheduler,
                                                                  initial_lr=initial_lr,
                                                                  normalization_mean=normalization_mean,
                                                                  normalization_std=normalization_std,
                                                                  clip_images=clip_images,
                                                                  reflection=reflection,
                                                                  eps=eps)

        self.device = get_working_device()
        # Image valid grid, each image value can only be 0 - 255 before normalization
        t = torch.from_numpy(np.array(list(range(256))).repeat(3).reshape(-1, 3) / 255)
        self.valid_grid = Normalize(mean=normalization_mean,
                                    std=normalization_std)(t.transpose(1, 0)[None, :, :, None]).squeeze().to(self.device)


        # Set the mean axis based on the image granularity
        if self.image_granularity == ImageGranularity.ImageWise:
            self.mean_axis = [H_AXIS, W_AXIS]
        else:
            self.mean_axis = [BATCH_AXIS, H_AXIS, W_AXIS]

        # Create BatchOptimizationHolder objects for each batch in the initial dataset
        self.batch_opt_holders_list = []
        for data_input in init_dataset:
            if isinstance(data_input, list):
                # This is the case in which the data loader holds both images and targets
                batched_images, targets = data_input
                targets.to(self.device)
            else:
                batched_images = data_input
                # targets = torch.randint(1000, [batched_images.size(0)])
            self.batch_opt_holders_list.append(
                PytorchBatchOptimizationHolder(
                    images=batched_images.to(self.device),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    initial_lr=initial_lr))
        self.n_batches = len(self.batch_opt_holders_list)
        self.random_batch_reorder()
        self.all_imgs_stats_holder = PytorchAllImagesStatsHolder(n_batches=self.n_batches,
                                                                 batch_size=self.batch_size,
                                                                 mean_axis=self.mean_axis)

        # Initialize statistics if using all data stats
        if self.use_all_data_stats:
            for i_batch in range(self.n_batches):
                input_imgs = self.image_pipeline.image_output_finalize(self.get_images_by_batch_index(i_batch))
                output = activation_extractor.run_model(input_imgs)
                self.all_imgs_stats_holder.update_batch_stats(batch_index=i_batch,
                                                              input_imgs=input_imgs,
                                                              activation_extractor=activation_extractor,
                                                              to_differentiate=False)

    def get_layer_accumulated_stats(self, layer_name: str) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get the accumulated activation statistics for a layer.

        Args:
            layer_name (str): the name of the layer.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: The averaged activation statistics (mean, variance, and standard deviation) on all the batches for the specified layer.
        """
        total_mean, total_second_moment = 0, 0
        for i_batch in range(self.n_batches):
            mean, second_moment, std = self.all_imgs_stats_holder.get_stats(i_batch, layer_name)
            total_mean += mean
            total_second_moment += second_moment

        total_mean /= self.n_batches
        total_second_moment /= self.n_batches
        total_var = total_second_moment - torch.pow(total_mean, 2)
        total_std = torch.sqrt(total_var + self.eps)
        return total_mean, total_std

    def optimization_step(self,
                          batch_index: int,
                          loss: Tensor,
                          i_ter: int):
        """
        Perform an optimization step.

        Args:
            batch_index (int): Index of the batch.
            loss (Tensor): Loss value.
            i_ter (int): Current optimization iteration.
        """
        # Get optimizer and scheduler for the specific batch index
        optimizer = self.get_optimizer_by_batch_index(batch_index)
        scheduler = self.get_scheduler_by_batch_index(batch_index)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Perform scheduler step
        self.scheduler_step_fn(scheduler, i_ter, loss.item())

        if self.clip_images:
            self.batch_opt_holders_list[batch_index].clip_images(self.valid_grid, reflection=self.reflection)


    def zero_grad(self, batch_index: int):
        """
        Zero the gradients for the specific batch index.

        Args:
            batch_index (int): Index of the batch.
        """
        # Get optimizer for the specific batch index
        optimizer = self.get_optimizer_by_batch_index(batch_index)

        # Zero gradients
        optimizer.zero_grad()
        self.model.zero_grad()


    def get_finalized_images(self) -> List[Tensor]:
        """
        Create and return a list of the optimized images.

        Returns:
            List[Tensor]: a list of the optimized images.
        """
        finalized_images = []

        # Iterate over each batch
        for i_batch in range(self.n_batches):
            # Retrieve the images for the current batch
            batch_imgs = self.get_images_by_batch_index(i_batch)

            # Apply the image_pipeline's image_output_finalize method to finalize the batch of images
            finalized_batch = self.image_pipeline.image_output_finalize(batch_imgs).detach().clone().cpu()

            # Split the finalized batch into individual images and add them to the finalized_images list
            finalized_images += torch.split(finalized_batch, 1)

        return finalized_images


class PytorchBatchOptimizationHolder(BatchOptimizationHolder):
    """
    An extension of BatchOptimizationHolder specifically for PyTorch models.
    Holds optimization parameters for a batch of images.

    This class acts as a container for optimization-related parameters specific to a batch of images. It does not
    directly manage or handle the optimization process itself but rather holds the necessary components for
    optimization, including images, optimizer and scheduler.
    """
    def __init__(self,
                 images: Tensor,
                 optimizer: Optimizer,
                 scheduler: Any,
                 initial_lr: float):
        """
        Constructor for the PytorchBatchOptimizationHolder class.

        Args:
            images (Tensor): A tensor containing the input images.
            optimizer (Optimizer): An optimizer responsible for updating the image parameters during optimization.
            scheduler (Any): A scheduler responsible for adjusting the learning rate of the optimizer over time.
            initial_lr (float): The initial learning rate used by the optimizer.
        """
        self.images = images
        self.images.requires_grad = True
        self.optimizer = optimizer([self.images], lr=initial_lr)
        self.scheduler = scheduler(self.optimizer)

    def clip_images(self,
                    valid_grid: Tensor,
                    reflection: bool = True):
        """
        Clip the images.

        Args:
            valid_grid (Tensor): A tensor containing valid values for image clipping.
            reflection (bool): Whether to use reflection during image clipping. Defaults to True.
        """
        with torch.no_grad():
            for i_ch in range(valid_grid.shape[0]):
                clamp = torch.clamp(self.images[:, i_ch, :, :], valid_grid[i_ch, :].min(), valid_grid[i_ch, :].max())
                if reflection:
                    self.images[:, i_ch, :, :] = 2 * clamp - self.images[:, i_ch, :, :]
                else:
                    self.images[:, i_ch, :, :] = clamp
        self.images.requires_grad = True


class PytorchAllImagesStatsHolder(AllImagesStatsHolder):
    """
    An extension of AllImagesStatsHolder specifically for PyTorch models.
    Stores activation statistics for all image batches. It offers an organized mechanism for retaining mean,
    second-moment, and standard deviation statistics corresponding to the activations of each layer.

    Responsible for collecting and storing activation statistics across all batches of images.
    It stores a list 'batches_stats_holder_list' of 'BatchStatsHolder's. Each `BatchStatsHolder` instance in
    the `batches_stats_holder_list` is responsible for storing statistics for a specific batch, specified by "batch_index".
    """
    def get_batches_stats_holder_list(self) -> List[BatchStatsHolder]:
        """
        Get a list of BatchStatsHolder objects.

        Returns:
            List[BatchStatsHolder]: A list of BatchStatsHolder objects.
        """
        return [PytorchBatchStatsHolder(self.mean_axis) for _ in range(self.n_batches)]



class PytorchBatchStatsHolder(BatchStatsHolder):
    """
    An extension of BatchStatsHolder specifically for PyTorch models.
    Stores activation statistics for a specific batch of images.
    This class provides a structured approach for managing mean, second-moment,
    and standard deviation statistics related to the activations of each layer
    for a particular batch of images.
    """
    def __init__(self,
                 mean_axis: List[int],
                 eps: float = 1e-6):
        """
        Constructor for the PytorchBatchStatsHolder class.

        Args:
            mean_axis (List[int]): The axis along which to compute the mean.
            eps (float): A small value added to the denominator to avoid division by zero. Defaults to 1e-6.
        """
        super(PytorchBatchStatsHolder, self).__init__(mean_axis=mean_axis, eps=eps)

    def get_var(self, bn_layer_name: str) -> Tensor:
        """
        Calculate the variance for the specified layer.

        Args:
            bn_layer_name (str): the name of the layer.

        Returns:
            Tensor: the variance for the specified layer.
        """
        mean = self.get_mean(bn_layer_name)
        second_moment = self.get_second_moment(bn_layer_name)
        var = second_moment - torch.pow(mean, 2.0)
        return var


    def get_std(self, bn_layer_name: str) -> Tensor:
        """
        Calculate the standard deviation for the specified layer.

        Args:
            bn_layer_name (str): the name of the layer.

        Returns:
            Tensor: The standard deviation for the specified layer.
        """
        var = self.get_var(bn_layer_name)
        return torch.sqrt(var + self.eps)

    def calc_bn_stats_from_activations(self,
                                       input_imgs: Tensor,
                                       activation_extractor: ActivationExtractor,
                                       to_differentiate: bool):
        """
        Calculate and update statistics (mean, second-moment) per layer, using the input images and activations.

        This function calculates and updates the mean and second-moment statistics for layers before batch normalization
        using the corresponding activations. The calculated statistics are used to align the batch normalization statistics
        stored in the original model during the data generation process.

        Args:
            input_imgs (Tensor): the input images tensor for which to calculate the statistics.
            activation_extractor (ActivationExtractor): the activation extractor object.
            to_differentiate (bool): a flag indicating whether to differentiate or not.
        """
        imgs_mean = torch.mean(input_imgs, dim=self.mean_axis)
        imgs_second_moment = torch.mean(torch.pow(input_imgs, 2.0), dim=self.mean_axis)
        if not to_differentiate:
            imgs_mean = imgs_mean.detach()
            imgs_second_moment = imgs_second_moment.detach()
        self.update_layer_stats(IMAGE_INPUT, imgs_mean, imgs_second_moment)
        # Extract statistics of intermediate convolution outputs before the BatchNorm layers
        for bn_layer_name in activation_extractor.get_extractor_layer_names():
            bn_input_activations = activation_extractor.get_layer_input_activation(bn_layer_name)
            if not to_differentiate:
                bn_input_activations = bn_input_activations.detach()

            collected_mean = torch.mean(bn_input_activations, dim=self.mean_axis)
            collected_second_moment = torch.mean(torch.pow(bn_input_activations, 2.0), dim=self.mean_axis)
            self.update_layer_stats(bn_layer_name, collected_mean, collected_second_moment)

    def clear(self):
        """Clear the statistics."""
        super().clear()
        torch.cuda.empty_cache()


class DatasetFromList(Dataset):
    """
    A custom Dataset that creates a Dataset from a list of images.
    """
    def __init__(self, img_list: List[Tensor]):
        """
        Constructor for the DatasetFromList class.

        Args:
            img_list (List[Tensor]): A list containing the images.
        """
        self.img_list = img_list

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The number of images in the dataset.
        """

        return len(self.img_list)

    def __getitem__(self, idx: int) -> Tensor:
        """
        Get the item at the specified index.

        Args:
            idx (int): the index of the item.

        Returns:
            Tensor: The image at the specified index.
        """
        return self.img_list[idx][0, :, :, :]