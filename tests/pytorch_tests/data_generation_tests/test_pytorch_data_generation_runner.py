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
import unittest

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from model_compression_toolkit.data_generation.common.enums import SchedulerType, BatchNormAlignemntLossType, \
    DataInitType, BNLayerWeightingType, ImageGranularity, ImagePipelineType, ImageNormalizationType, OutputLossType
from model_compression_toolkit.data_generation.pytorch.optimization_functions.lr_scheduler import \
    ReduceLROnPlateauWithReset
from tests.pytorch_tests.data_generation_tests.base_pytorch_data_generation_test import BasePytorchDataGenerationTest


class PytorchDataGenerationTestRunner(unittest.TestCase):
    def test_pytorch_scheduler_types(self):
        BasePytorchDataGenerationTest(self, scheduler=StepLR, scheduler_type=SchedulerType.STEP).run_test()
        BasePytorchDataGenerationTest(self, scheduler=ReduceLROnPlateau, scheduler_type=SchedulerType.REDUCE_ON_PLATEAU).run_test()
        BasePytorchDataGenerationTest(self, scheduler=ReduceLROnPlateauWithReset, scheduler_type=SchedulerType.REDUCE_ON_PLATEAU_WITH_RESET).run_test()

    def test_pytorch_layer_weighting_types(self):
        BasePytorchDataGenerationTest(self, layer_weighting_type=BNLayerWeightingType.AVERAGE).run_test()
        BasePytorchDataGenerationTest(self, layer_weighting_type=BNLayerWeightingType.FIRST_LAYER_MULTIPLIER).run_test()

    def test_pytorch_bn_alignment_types(self):
        BasePytorchDataGenerationTest(self, bn_alignment_loss_type=BatchNormAlignemntLossType.L2_SQUARE).run_test()

    def test_pytorch_data_init_types(self):
        BasePytorchDataGenerationTest(self, data_init_type=DataInitType.Gaussian).run_test()
        BasePytorchDataGenerationTest(self, data_init_type=DataInitType.Diverse).run_test()

    def test_pytorch_image_granularity_types(self):
        BasePytorchDataGenerationTest(self, image_granularity=ImageGranularity.ImageWise).run_test()
        BasePytorchDataGenerationTest(self, image_granularity=ImageGranularity.BatchWise).run_test()
        BasePytorchDataGenerationTest(self, image_granularity=ImageGranularity.AllImages).run_test()

    def test_pytorch_image_pipeline_types(self):
        BasePytorchDataGenerationTest(self, image_pipeline_type=ImagePipelineType.IDENTITY).run_test()
        BasePytorchDataGenerationTest(self, image_pipeline_type=ImagePipelineType.SMOOTHING_AND_AUGMENTATION, output_image_size=(32,), extra_pixels=32).run_test()
        BasePytorchDataGenerationTest(self, image_pipeline_type=ImagePipelineType.SMOOTHING_AND_AUGMENTATION, output_image_size=32, extra_pixels=(0,)).run_test()
        BasePytorchDataGenerationTest(self, image_pipeline_type=ImagePipelineType.SMOOTHING_AND_AUGMENTATION, image_clipping=True, output_image_size=(32, 32), extra_pixels=1).run_test()
        BasePytorchDataGenerationTest(self, image_pipeline_type=ImagePipelineType.SMOOTHING_AND_AUGMENTATION, image_clipping=False, output_image_size=(32, 24), extra_pixels=(1, 3)).run_test()

    def test_pytorch_image_normalization_types(self):
        BasePytorchDataGenerationTest(self, image_normalization_type=ImageNormalizationType.TORCHVISION).run_test()
        BasePytorchDataGenerationTest(self, image_normalization_type=ImageNormalizationType.NO_NORMALIZATION).run_test()

    def test_pytorch_output_loss_types(self):
        BasePytorchDataGenerationTest(self, output_loss_type=OutputLossType.NONE).run_test()
        BasePytorchDataGenerationTest(self, output_loss_type=OutputLossType.NEGATIVE_MIN_MAX_DIFF, output_loss_multiplier=0.1).run_test()
        BasePytorchDataGenerationTest(self, output_loss_type=OutputLossType.INVERSE_MIN_MAX_DIFF, output_loss_multiplier=0.1).run_test()
        BasePytorchDataGenerationTest(self, output_loss_type=OutputLossType.REGULARIZED_MIN_MAX_DIFF, output_loss_multiplier=0.1).run_test()

if __name__ == '__main__':
    unittest.main()