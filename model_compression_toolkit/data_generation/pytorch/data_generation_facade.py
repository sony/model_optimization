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
from functools import partial
from typing import Callable, Any, Tuple, List
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.constants import FOUND_TORCH
from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.constants import PYTORCH
from model_compression_toolkit.core.pytorch.utils import set_model
from model_compression_toolkit.data_generation.common.data_generation_config import DataGenerationConfig
from model_compression_toolkit.data_generation.pytorch.image_pipeline import PytorchImagePipeline, get_random_data
from model_compression_toolkit.data_generation.pytorch.model_info_exctractors import ActivationExtractor, \
    OrigBNStatsHolder
from model_compression_toolkit.data_generation.pytorch.optimization_utils import DataHolder

DEFAULT_INITIAL_LR = 5
DEFAULT_DATA_GEN_BS = 8


if FOUND_TORCH:
    import torch
    from torch.nn import Module
    from torch.optim import RAdam, Optimizer
    from torch.optim.lr_scheduler import LRScheduler, StepLR


    def get_pytorch_data_generation_config(
            n_iter: int,
            optimizer: Optimizer = RAdam,
            scheduler: LRScheduler = partial(StepLR, step_size=500),
            data_gen_batch_size=DEFAULT_DATA_GEN_BS,
            initial_lr=DEFAULT_INITIAL_LR,
            bna_loss_fn: Callable = None,
            image_pipeline: Callable = None,
            bn_layer_types: List = [torch.nn.BatchNorm2d],
            image_initialization_fn: Callable = None) -> DataGenerationConfig:
        """
        Create a DataGenerationConfig instance for Pytorch models.

        args:
            n_epochs (int): Number of epochs for running the representative dataset for fine-tuning.
            optimizer (Optimizer): Pytorch optimizer to use for fine-tuning for auxiliry variable.
            optimizer_rest (Optimizer): Pytorch optimizer to use for fine-tuning of the bias variable.
            loss (Callable): loss to use during fine-tuning. should accept 4 lists of tensors. 1st list of quantized tensors, the 2nd list is the float tensors, the 3rd is a list of quantized weights and the 4th is a list of float weights.
            log_function (Callable): Function to log information about the gptq process.

        returns:
            a DataGenerationConfig object to use when fine-tuning the quantized model using gptq.

        Examples:

            Import MCT and create a DataGenerationConfig with 500 iterations:

            >>> import model_compression_toolkit as mct
            >>> data_gen_conf = mct.data_generation.get_pytorch_data_generation_config(n_iter=500)

            Other PyTorch optimizers and schedulers can be used:

            >>> import torch
            >>> data_gen_conf = mct.data_generation.get_pytorch_data_generation_config(n_iter=100, optimizer=torch.optim.Adam, scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau)

        """
        return DataGenerationConfig(
            n_iter=n_iter,
            optimizer=optimizer,
            scheduler=scheduler,
            data_gen_batch_size=data_gen_batch_size,
            initial_lr=initial_lr,
            bna_loss_fn=bna_loss_fn,
            image_pipeline=image_pipeline,
            bn_layer_types=bn_layer_types,
            image_initialization_fn=image_initialization_fn)


    def pytorch_data_generation_experimental(
            model: Module,
            n_images: int,
            output_image_size: Tuple,
            data_generation_config: DataGenerationConfig):

        set_model(model)
        image_pipeline = data_generation_config.image_pipeline(output_image_size)
        activation_extractor = ActivationExtractor(model, data_generation_config.bn_layer_types)
        orig_bn_stats_holder = OrigBNStatsHolder(model)

        num_batches, init_dataset = data_generation_config.image_initialization_fn(
            n_images=n_images,
            image_size=image_pipeline.get_image_input_size(),
            batch_size=data_generation_config.data_gen_batch_size)

        data_holder = DataHolder(
            init_dataset,
            num_batches,
            model,
            data_generation_config,
            activation_extractor)

        for i_ter in range(data_generation_config.n_iter):
            for i_batch in range(num_batches):
                imgs_to_optimize = data_holder.get_batched_images_opt(i_batch)
                data_holder.optimizer.zero_grad()
                model.zero_grad()
                imgs_input = image_pipeline.image_manipulation(imgs_to_optimize)
                output = model(imgs_input)
                bn_loss = data_holder.bn_loss(imgs_input, i_batch, activation_extractor, orig_bn_stats_holder, data_generation_config)
                other_losses = data_generation_config.activation_losses(imgs_input, output, activation_extractor)
                total_loss = bn_loss + other_losses
                total_loss.backward()
                data_holder.scheduler_step(total_loss.item())
                data_holder.optimizer.step()
                data_holder.update_statistics(i_batch, activation_extractor, to_differentiate=False)
        return data_holder.get_finilized_data_loader()
else:
    # If torch is not installed,
    # we raise an exception when trying to use these functions.
    def get_pytorch_data_generation_config(*args, **kwargs):
        Logger.critical('Installing Pytorch is mandatory '
                        'when using get_pytorch_data_generation_config. '
                        'Could not find torch package.')  # pragma: no cover


    def pytorch_data_generation_experimental(*args, **kwargs):
        Logger.critical('Installing Pytorch is mandatory '
                        'when using pytorch_data_generation_experimental. '
                        'Could not find the torch package.')  # pragma: no cover


if __name__ == '__main__':
    import torchvision
    from torchvision.models import ResNet18_Weights

    def f_norm(a, b):
        return torch.linalg.norm(a - b) ** 2 / b.size(0)

    data_generation_config = get_pytorch_data_generation_config(
        n_iter=10,
        bna_loss_fn=f_norm,
        image_pipeline=PytorchImagePipeline,
        bn_layer_types=[torch.nn.BatchNorm2d],
    image_initialization_fn=get_random_data)
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    pytorch_data_generation_experimental(model,
                                         n_images=64,
                                         output_image_size=224,
                                         data_generation_config=data_generation_config)
