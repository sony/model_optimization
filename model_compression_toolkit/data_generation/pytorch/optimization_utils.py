from typing import List, Callable, Type, Any, Dict, Tuple
import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from model_compression_toolkit.data_generation.common.data_generation_config import DataGenerationConfig, \
    ImageGranularity, BaseImagePipeline
from model_compression_toolkit.data_generation.pytorch.constants import DEVICE, IMAGE_INPUT, BATCH_AXIS, H_AXIS, W_AXIS
from model_compression_toolkit.data_generation.pytorch.model_info_exctractors import ActivationExtractor, \
    OrigBNStatsHolder


class AllImagesOptimizationHandler:
    def __init__(self,
                 init_dataset: DataLoader,
                 data_generation_config: DataGenerationConfig,
                 image_pipeline: BaseImagePipeline,
                 activation_extractor: ActivationExtractor,
                 eps: float = 1e-6):
        """
        Constructor for the AllImagesOptimizationHandler class.

        Args:
            init_dataset (torch.utils.data.Dataset): The initial dataset used for images generation.
            data_generation_config (DataGenerationConfig): Configurations for data generation.
            image_pipeline (BaseImagePipeline): The image pipeline for processing images.
            activation_extractor (ActivationExtractor): Extractor for layer activations.
            eps (float): A small value added for numerical stability.
        """
        self.data_generation_config = data_generation_config
        self.image_pipeline = image_pipeline
        self.batch_size = self.data_generation_config.data_gen_batch_size
        self.eps = eps
        self.targets = []
        self.use_all_data_stats = False

        # Determine if all data statistics should be used
        if data_generation_config.image_granularity in [ImageGranularity.AllImages]:
            self.use_all_data_stats = True

        # Set the mean axis based on the image granularity
        self.mean_axis = data_generation_config.get_dimensions_for_average()

        # Create BatchOptimizationHolder objects for each batch in the initial dataset
        self.batch_opt_holders_list = []
        for data_input in init_dataset:
            if isinstance(data_input, list):
                # This is the case in which the data loader holds both images and targets
                batched_images, targets = data_input
                targets.to(DEVICE)
            else:
                batched_images = data_input
                targets = None
            self.batch_opt_holders_list.append(
                BatchOptimizationHolder(
                    images=batched_images.to(DEVICE),
                    optimizer=data_generation_config.optimizer,
                    scheduler=data_generation_config.scheduler,
                    initial_lr=data_generation_config.initial_lr))
            self.targets.append(targets)
        self.n_batches = len(self.batch_opt_holders_list)
        self.random_batch_reorder()
        self.all_imgs_stats_holder = AllImgsStatsHolder(n_batches=self.n_batches,
                                                        batch_size=self.batch_size,
                                                        mean_axis=self.mean_axis)

        # Initialize statistics if using all data stats
        if self.use_all_data_stats:
            for i_batch in range(self.n_batches):
                input_imgs = self.image_pipeline.image_output_finalize(self.get_images_by_batch_index(i_batch))
                output = activation_extractor.run_on_inputs(input_imgs)
                self.all_imgs_stats_holder.update_batch_stats(batch_index=i_batch,
                                                              input_imgs=input_imgs,
                                                              activation_extractor=activation_extractor,
                                                              to_differentiate=False)

    def random_batch_reorder(self):
        """
        Randomly reorders the batch indices.
        """
        self.rand_batch_inds = np.random.choice(self.n_batches, self.n_batches, replace=False)

    def get_random_batch_index(self, index: int) -> int:
        """
        Get the random batch index at the specified index.

         Args:
            index (int): The index.

        Returns:
            int: The random batch index.
        """
        return self.rand_batch_inds[index]

    def get_images_by_batch_index(self, batch_index: int) -> Tensor:
        """
        Retrieves the images from the batch optimization holder at the given batch index.

        Args:
            batch_index (int): The index of the batch optimization holder.

        Returns:
            Tensor: The images in the batch.
        """
        return self.batch_opt_holders_list[batch_index].get_images()

    def get_optimizer_by_batch_index(self, batch_index: int) -> Optimizer:
        """
        Get the optimizer for the specific batch specified by the batch index.

        Args:
            batch_index (int) : the index of the batch.

        Returns:
            Optimizer: the optimizer of the specific batch specified by the batch index.
        """
        return self.batch_opt_holders_list[batch_index].get_optimizer()

    def get_scheduler_by_batch_index(self, batch_index: int):
        """
        Get the scheduler for the specific batch specified by the batch index.

        Args:
            batch_index (str): the index of the batch.

        Returns:
            LRScheduler: the scheduler of the specific batch specified by the batch index.
        """
        return self.batch_opt_holders_list[batch_index].get_scheduler()

    def get_accumulated_stats_per_layer(self, layer_name: str) -> Tuple[Tensor, Tensor]:
        """
        Get the accumulated activation statistics for a layer.

        Args:
            layer_name (str): the name of the layer.

        Returns:
            Tuple[Tensor, Tensor]: the averaged activation statistics on all the batches for the specified layer.
        """
        total_mean, total_second_moment = 0, 0
        for i_batch in range(self.n_batches):
            mean, second_moment = self.all_imgs_stats_holder.get_stats(i_batch, layer_name)
            total_mean += mean
            total_second_moment += second_moment

        total_mean /= self.n_batches
        total_second_moment /= self.n_batches
        total_var = total_second_moment - torch.pow(total_mean, 2)
        return total_mean, total_var

    def compute_bn_loss(self,
                        input_imgs: Tensor,
                        batch_index: int,
                        activation_extractor: ActivationExtractor,
                        orig_bn_stats_holder: OrigBNStatsHolder,
                        bn_loss_fn: Callable,
                        layer_weights: Dict) -> Tensor:
        """
        Compute the batch norm alignment loss.

        Args:
            input_imgs (Tensor): the input images.
            batch_index (int): the index of the batch.
            activation_extractor (ActivationExtractor): extractor for layer activations.
            orig_bn_stats_holder (OrigBNStatsHolder): holder for original BatchNorm statistics.
            bn_loss_fn (Callable): the batch norm alignment loss function.
            layer_weights (Dict): weights to multiply the loss for each layer.

        Returns:
            Tensor: the computed batch norm alignment loss.
        """
        # Update the batch statistics for the current batch
        self.all_imgs_stats_holder.update_batch_stats(batch_index=batch_index,
                                                      input_imgs=input_imgs,
                                                      activation_extractor=activation_extractor,
                                                      to_differentiate=True)

        # Initialize variables for accumulating mean and variance differences
        total_mean_diff, total_var_diff = 0, 0

        # Iterate over each BN layer
        for layer_name in orig_bn_stats_holder.get_bn_layer_names():
            # Get the layer weight for the current BN layer
            layer_weight = layer_weights.get(layer_name)

            # Get the mean and variance from the original BN statistics
            bn_layer_mean = orig_bn_stats_holder.get_mean(layer_name)
            bn_layer_var = orig_bn_stats_holder.get_var(layer_name)

            # Get the mean and variance from the current batch's statistics
            if self.use_all_data_stats:
                # If using all data statistics, retrieve the accumulated statistics from all batches
                imgs_layer_mean, imgs_layer_var = self.get_accumulated_stats_per_layer(layer_name)
            else:
                # Otherwise, retrieve the statistics from the current batch
                imgs_layer_mean, imgs_layer_var = self.all_imgs_stats_holder.get_stats(batch_index, layer_name)

            # Calculate the standard deviation from the variance
            bn_layer_std = torch.sqrt(bn_layer_var + self.eps)
            imgs_layer_std = torch.sqrt(imgs_layer_var + self.eps)

            # Accumulate the mean and variance loss metrics weighted by the layer weight
            total_mean_diff += layer_weight * bn_loss_fn(bn_layer_mean, imgs_layer_mean)
            total_var_diff += layer_weight * bn_loss_fn(bn_layer_std, imgs_layer_std)

        # Compute the total BN loss as the sum of mean and variance differences
        total_bn_loss = total_mean_diff + total_var_diff

        return total_bn_loss

    def update_statistics(self,
                        input_imgs: Tensor,
                        batch_index: int,
                        activation_extractor: ActivationExtractor):
        """
        Update the statistics for the images at the specified batch index.

        Args:
            input_imgs (Tensor): the input images.
            batch_index (int): the index of the batch.
            activation_extractor (ActivationExtractor): extractor for layer activations.
        """
        if self.use_all_data_stats:
            self.all_imgs_stats_holder.update_batch_stats(batch_index=batch_index,
                                                          input_imgs=input_imgs,
                                                          activation_extractor=activation_extractor,
                                                          to_differentiate=False)

    def get_finalized_data_loader(self) -> DataLoader:
        """
        Create and return a DataLoader using the optimized images.

        Returns:
            DataLoader: a DataLoader object using the optimized images.
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

        # Create a DatasetFromList object using the finalized_images list
        tensor_dataset = DatasetFromList(finalized_images)

        # Create and return a DataLoader using the tensor_dataset
        return DataLoader(
            tensor_dataset,
            batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=True)


class BatchOptimizationHolder:

    def __init__(self,
                 images: Tensor,
                 optimizer: Optimizer,
                 scheduler: Any,
                 initial_lr: float):
        """
        Constructor for the BatchOptimizationHolder class.

        Args:
            images (Tensor): a tensor containing the input images.
            optimizer (Optimizer): optimizer responsible for updating the image parameters during optimization.
            scheduler (Any): scheduler responsible for adjusting the learning rate of the optimizer over time.
            initial_lr (float): the initial learning rate used by the optimizer.
        """
        self.images = images
        self.images.requires_grad = True
        self.optimizer = optimizer([self.images], lr=initial_lr)
        self.scheduler = scheduler(self.optimizer)

    def get_images(self):
        """Returns the stored images"""
        return self.images

    def get_optimizer(self):
        """Returns the optimizer"""
        return self.optimizer

    def get_scheduler(self):
        "Returns the scheduler"
        return self.scheduler


class AllImgsStatsHolder:
    def __init__(self,
                 n_batches: int,
                 batch_size: int,
                 mean_axis=Type[list]):
        """
        Constructor for the AllImgsStatsHolder class.

        Args:
            n_batches (int): the number of batches.
            batch_size (int): the size of each batch.
            mean_axis (int): the axis along which to compute the mean.
        """
        self.mean_axis = mean_axis
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.batches_stats_holder_list = [BatchStatsHolder(self.mean_axis) for _ in range(self.n_batches)]
        self.data_bn_mean_all_batches_except_one = {}
        self.data_bn_second_moment_all_batches_except_one = {}
        self.bn_mean_all_batches = {}
        self.bn_second_moment_all_batches = {}

    def update_batch_stats(self,
                           batch_index: int,
                           input_imgs: Tensor,
                           activation_extractor: ActivationExtractor,
                           to_differentiate=False):
        """
        Update the batch statistics for a given batch.

        Args:
            batch_index (int): the index of the batch.
            input_imgs (Tensor): the input images for which to calculate the statistics.
            activation_extractor (ActivationsExtractor): the activation extractor object.
            to_differentiate (bool): a flag indicating whether to differentiate or not. Defaults to False.
        """
        self.batches_stats_holder_list[batch_index].clear()
        self.batches_stats_holder_list[batch_index].calc_bn_stats_from_activations(input_imgs=input_imgs,
                                                                                   activation_extractor=activation_extractor,
                                                                                   to_differentiate=to_differentiate)

    def get_stats(self,
                  batch_index: int,
                  layer_name: str) -> Tuple[Tensor, Tensor]:
        """
        Get the statistics for a given batch and layer.

        Args:
            batch_index (int): the index of the batch.
            layer_name (str): the name of the layer.

        Returns:
            Tuple[Tensor, Tensor]: the mean and second moment for the specified batch and layer.
        """
        mean = self.batches_stats_holder_list[batch_index].get_mean(layer_name)
        second_moment = self.batches_stats_holder_list[batch_index].get_second_moment(layer_name)
        return mean, second_moment


class BatchStatsHolder(object):

    def __init__(self,
                 mean_axis: Type[list],
                 eps: float = 1e-6):
        """
        Constructor for the BatchStatsHolder class.

        Args:
            mean_axis (Type[list]): the axis along which to compute the mean.
            eps (float): a small value added to the denominator to avoid division by zero. Defaults to 1e-6.
        """
        self.eps = eps
        self.mean_axis = mean_axis
        self.bn_mean = {}
        self.bn_second_moment = {}

    def get_mean(self, bn_layer_name: str) -> Tensor:
        """
        Get the mean for the specified layer.

        Args:
            bn_layer_name (str): the name of the layer.

        Returns:
            Tensor: the mean for the specified layer.
        """
        return self.bn_mean[bn_layer_name]

    def get_second_moment(self, bn_layer_name: str) -> Tensor:
        """
        Get the second moment for the specified layer.

        Args:
            bn_layer_name (str): the name of the layer.

        Returns:
            Tensor: the second moment for the specified layer.
        """
        return self.bn_second_moment[bn_layer_name]

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

    def update_layer_stats(self,
                           bn_layer_name: str,
                           mean: Tensor,
                           second_moment: Tensor):
        """
        Update the statistics for a layer.

        Args:
            bn_layer_name (str): the name of the layer.
            mean (Tensor): the mean value for the layer.
            second_moment (Tensor): the second moment value for the layer.
        """
        self.bn_mean.update({bn_layer_name: mean})
        self.bn_second_moment.update({bn_layer_name: second_moment})

    def calc_bn_stats_from_activations(self,
                                       input_imgs: Tensor,
                                       activation_extractor: ActivationExtractor,
                                       to_differentiate: bool):
        """
        Calculate statistics from the input images and activation extractor.

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
        for bn_layer_name in activation_extractor.get_bn_layer_names():
            bn_input_activations = activation_extractor.get_activation(bn_layer_name)
            if not to_differentiate:
                bn_input_activations = bn_input_activations.detach()

            collected_mean = torch.mean(bn_input_activations, dim=self.mean_axis)
            collected_second_moment = torch.mean(torch.pow(bn_input_activations, 2.0), dim=self.mean_axis)
            self.update_layer_stats(bn_layer_name, collected_mean, collected_second_moment)

    def clear(self):
        """Clear the statistics."""
        self.bn_mean.clear()
        self.bn_second_moment.clear()
        self.bn_mean = {}
        self.bn_second_moment = {}
        torch.cuda.empty_cache()


class DatasetFromList(Dataset):

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
            The number of images in the dataset.
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