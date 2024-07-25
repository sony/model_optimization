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

from typing import List

import keras
from keras.layers import ReLU
from tensorflow.keras.optimizers.legacy import Optimizer, Adam
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Dense, GlobalAveragePooling2D

from model_compression_toolkit.data_generation import keras_data_generation_experimental, \
    get_keras_data_generation_config
from model_compression_toolkit.data_generation.common.enums import (SchedulerType,
                                                                    BatchNormAlignemntLossType, OutputLossType,
                                                                    DataInitType, BNLayerWeightingType,
                                                                    ImageGranularity,
                                                                    ImagePipelineType,
                                                                    ImageNormalizationType)


def DataGenerationModel():
    input_shape = (32, 32, 3)
    inputs = Input(shape=input_shape)
    x = Conv2D(3, 3)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(3, 3)(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = ReLU()(x)
    x = Dense(30)(x)
    return keras.Model(inputs=inputs, outputs=x)

def NoBNDataGenerationModel():
    input_shape = (32, 32, 3)
    inputs = Input(shape=input_shape)
    x = Conv2D(3, 3)(inputs)
    return keras.Model(inputs=inputs, outputs=x)

class BaseKerasDataGenerationTest:
    def __init__(self,
                 unit_test,
                 model: keras.Model = None,
                 n_images: int = 32,
                 output_image_size: int = (32, 32),
                 n_iter: int = 10,
                 optimizer: Optimizer = Adam,
                 scheduler_type: SchedulerType = SchedulerType.REDUCE_ON_PLATEAU,
                 data_gen_batch_size=8,
                 initial_lr=1.0,
                 output_loss_multiplier=0.0001,
                 bn_alignment_loss_type: BatchNormAlignemntLossType = BatchNormAlignemntLossType.L2_SQUARE,
                 output_loss_type: OutputLossType = OutputLossType.NONE,
                 data_init_type: DataInitType = DataInitType.Gaussian,
                 layer_weighting_type: BNLayerWeightingType = BNLayerWeightingType.AVERAGE,
                 image_granularity=ImageGranularity.BatchWise,
                 image_pipeline_type: ImagePipelineType = ImagePipelineType.SMOOTHING_AND_AUGMENTATION,
                 image_normalization_type: ImageNormalizationType = ImageNormalizationType.KERAS_APPLICATIONS,
                 extra_pixels: int = 0,
                 image_clipping: bool = False,
                 bn_layer_types: List = [BatchNormalization]
                 ):
        self.unit_test = unit_test
        self.model = model
        if model is None:
            self.model = DataGenerationModel()
        self.n_images = n_images
        self.output_image_size = output_image_size
        self.n_iter = n_iter
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
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

    def run_test(self):
        data_generation_config = get_keras_data_generation_config(
            n_iter=self.n_iter,
            data_gen_batch_size=self.data_gen_batch_size,
            initial_lr=self.initial_lr,
            optimizer=self.optimizer,
            scheduler_type=self.scheduler_type,
            image_normalization_type=self.image_normalization_type,
            image_pipeline_type=self.image_pipeline_type,
            layer_weighting_type=self.layer_weighting_type,
            image_granularity=self.image_granularity,
            data_init_type=self.data_init_type,
            extra_pixels=self.extra_pixels,
            image_clipping=self.image_clipping,
            output_loss_type=self.output_loss_type,
            output_loss_multiplier=self.output_loss_multiplier)

        data_loader = keras_data_generation_experimental(
            model=self.model,
            n_images=self.n_images,
            output_image_size=self.output_image_size,
            data_generation_config=data_generation_config)
