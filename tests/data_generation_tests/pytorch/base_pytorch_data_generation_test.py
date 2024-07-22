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
from typing import Any, List

import torch
from torch import nn
from torch.optim import Optimizer, RAdam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

from model_compression_toolkit.data_generation.common.data_generation_config import DataGenerationConfig
from model_compression_toolkit.data_generation.common.enums import SchedulerType, BatchNormAlignemntLossType, \
    DataInitType, BNLayerWeightingType, ImageGranularity, ImagePipelineType, ImageNormalizationType, OutputLossType
from model_compression_toolkit.data_generation.pytorch.pytorch_data_generation import \
    pytorch_data_generation_experimental


class BaseDataGenerationModel(torch.nn.Module):
    def __init__(self):
        super(BaseDataGenerationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        out = self.conv3(out)
        return out

class BasePytorchDataGenerationTest:

    def __init__(self,
                 unit_test,
                 n_images: int = 32,
                 output_image_size: int = 32,
                 n_iter: int = 10,
                 optimizer: Optimizer = RAdam,
                 scheduler: Any = partial(StepLR, step_size=10),
                 data_gen_batch_size=8,
                 initial_lr=0.05,
                 output_loss_multiplier=0.0001,
                 scheduler_type: SchedulerType = SchedulerType.STEP,
                 bn_alignment_loss_type: BatchNormAlignemntLossType = BatchNormAlignemntLossType.L2_SQUARE,
                 output_loss_type: OutputLossType = OutputLossType.NONE,
                 data_init_type: DataInitType = DataInitType.Diverse,
                 layer_weighting_type: BNLayerWeightingType = BNLayerWeightingType.AVERAGE,
                 image_granularity=ImageGranularity.AllImages,
                 image_pipeline_type: ImagePipelineType = ImagePipelineType.SMOOTHING_AND_AUGMENTATION,
                 image_normalization_type: ImageNormalizationType = ImageNormalizationType.TORCHVISION,
                 extra_pixels: int = 0,
                 image_clipping: bool = True,
                 bn_layer_types: List = [torch.nn.BatchNorm2d]
                 ):
        self.unit_test = unit_test
        self.model = BaseDataGenerationModel()
        self.n_images = n_images
        self.output_image_size = output_image_size
        self.n_iter = n_iter
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_gen_batch_size = data_gen_batch_size
        self.initial_lr = initial_lr
        self.output_loss_multiplier = output_loss_multiplier
        self.scheduler_type = scheduler_type
        self.bn_alignment_loss_type = bn_alignment_loss_type
        self.output_loss_type = output_loss_type
        self.data_init_type = data_init_type
        self.layer_weighting_type = layer_weighting_type
        self.image_granularity = image_granularity
        self.image_pipeline_type = image_pipeline_type
        self.image_normalization_type = image_normalization_type
        self.extra_pixels = extra_pixels
        self.image_clipping = image_clipping
        self.bn_layer_types = bn_layer_types


    def get_data_generation_config(self):
        return DataGenerationConfig(
            n_iter=self.n_iter,
            optimizer=self.optimizer,
            data_gen_batch_size=self.data_gen_batch_size,
            initial_lr=self.initial_lr,
            output_loss_multiplier=self.output_loss_multiplier,
            scheduler_type=self.scheduler_type,
            bn_alignment_loss_type=self.bn_alignment_loss_type,
            output_loss_type=self.output_loss_type,
            data_init_type=self.data_init_type,
            layer_weighting_type=self.layer_weighting_type,
            image_granularity=self.image_granularity,
            image_pipeline_type=self.image_pipeline_type,
            image_normalization_type=self.image_normalization_type,
            extra_pixels=self.extra_pixels,
            image_clipping=self.image_clipping,
            bn_layer_types=self.bn_layer_types)

    def run_test(self):
        data_generation_config = self.get_data_generation_config()
        data_loader = pytorch_data_generation_experimental(
            model=self.model,
            n_images=self.n_images,
            output_image_size=self.output_image_size,
            data_generation_config=data_generation_config)
