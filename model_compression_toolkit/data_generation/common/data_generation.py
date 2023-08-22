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
# Import required modules and classes
import time
from typing import Callable, Any, List

import torch
from tqdm import tqdm

from model_compression_toolkit.core.pytorch.utils import get_working_device
from model_compression_toolkit.data_generation.common.data_generation_config import DataGenerationConfig
from model_compression_toolkit.data_generation.common.image_pipeline import BaseImagePipeline
from model_compression_toolkit.data_generation.common.model_info_exctractors import ActivationExtractor, \
    OriginalBNStatsHolder
from model_compression_toolkit.data_generation.common.optimization_utils import ImagesOptimizationHandler
from model_compression_toolkit.logger import Logger


def data_generation(
        data_generation_config: DataGenerationConfig,
        activation_extractor: ActivationExtractor,
        orig_bn_stats_holder: OriginalBNStatsHolder,
        all_imgs_opt_handler: ImagesOptimizationHandler,
        image_pipeline: BaseImagePipeline,
        bn_layer_weighting_fn: Callable,
        bn_alignment_loss_fn: Callable,
        output_loss_fn: Callable,
        output_loss_multiplier: float
) -> List[Any]:
    """
    Function to perform data generation using the provided model and data generation configuration.

    Args:
        data_generation_config (DataGenerationConfig): Configuration for data generation.
        activation_extractor (ActivationExtractor): The activation extractor for the model.
        orig_bn_stats_holder (OriginalBNStatsHolder): Object to hold original BatchNorm statistics.
        all_imgs_opt_handler (ImagesOptimizationHandler): Handles the images optimization process.
        image_pipeline (Callable): Callable image pipeline for image manipulation.
        bn_layer_weighting_fn (Callable): Function to compute layer weighting for the BatchNorm alignment loss .
        bn_alignment_loss_fn (Callable): Function to compute BatchNorm alignment loss.
        output_loss_fn (Callable): Function to compute output loss.
        output_loss_multiplier (float): Multiplier for the output loss.

    Returns:
        List: Finalized list containing generated images.
    """

    # Compute the layer weights based on orig_bn_stats_holder
    bn_layer_weights = bn_layer_weighting_fn(orig_bn_stats_holder)

    # Get the current time to measure the total time taken
    total_time = time.time()

    # Create a tqdm progress bar for iterating over data_generation_config.n_iter iterations
    ibar = tqdm(range(data_generation_config.n_iter))

    # Perform data generation iterations
    for i_ter in ibar:

        # Randomly reorder the batches
        all_imgs_opt_handler.random_batch_reorder()

        # Iterate over each batch
        for i_batch in range(all_imgs_opt_handler.n_batches):
            # Get the random batch index
            random_batch_index = all_imgs_opt_handler.get_random_batch_index(i_batch)

            # Get the images to optimize and the optimizer for the batch
            imgs_to_optimize = all_imgs_opt_handler.get_images_by_batch_index(random_batch_index)

            # Zero gradients
            all_imgs_opt_handler.zero_grad(random_batch_index)

            # Perform image input manipulation
            input_imgs = image_pipeline.image_input_manipulation(imgs_to_optimize)

            # Forward pass to extract activations
            output = activation_extractor.run_model(input_imgs)

            # Compute BatchNorm alignment loss
            bn_loss = all_imgs_opt_handler.compute_bn_loss(input_imgs=input_imgs,
                                                           batch_index=random_batch_index,
                                                           activation_extractor=activation_extractor,
                                                           orig_bn_stats_holder=orig_bn_stats_holder,
                                                           bn_alignment_loss_fn=bn_alignment_loss_fn,
                                                           bn_layer_weights=bn_layer_weights)


            # Compute output loss
            output_loss = output_loss_fn(output_imgs=output) if output_loss_multiplier > 0 else torch.zeros(1).to(get_working_device())

            # Compute total loss
            total_loss = bn_loss + output_loss_multiplier * output_loss

            # Perform optimiztion step
            all_imgs_opt_handler.optimization_step(random_batch_index, total_loss, i_ter)

            # Update the statistics based on the updated images
            if all_imgs_opt_handler.use_all_data_stats:
                final_imgs = image_pipeline.image_output_finalize(imgs_to_optimize)
                all_imgs_opt_handler.update_statistics(input_imgs=final_imgs,
                                                       batch_index=random_batch_index,
                                                       activation_extractor=activation_extractor)

        ibar.set_description(f"Total Loss: {total_loss.item():.5f}, "
                            f"BN Loss: {bn_loss.item():.5f}, "
                            f"Output Loss: {output_loss.item():.5f}")


    # Return a list containing the finalized generated images
    finalized_imgs = all_imgs_opt_handler.get_finalized_images()
    Logger.info(f'Total time to generate {len(finalized_imgs)} images (seconds): {int(time.time() - total_time)}')
    return finalized_imgs
