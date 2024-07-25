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

from model_compression_toolkit.data_generation.common.enums import SchedulerType, BatchNormAlignemntLossType, \
    DataInitType, BNLayerWeightingType, ImageGranularity, ImagePipelineType, ImageNormalizationType, OutputLossType
from tests.data_generation_tests.keras.base_keras_data_generation_test import BaseKerasDataGenerationTest, \
    NoBNDataGenerationModel


class KerasDataGenerationTestRunner(unittest.TestCase):
    def test_keras_scheduler_types(self):
        BaseKerasDataGenerationTest(self, scheduler_type=SchedulerType.REDUCE_ON_PLATEAU).run_test()

    def test_keras_layer_weighting_types(self):
        BaseKerasDataGenerationTest(self, layer_weighting_type=BNLayerWeightingType.AVERAGE).run_test()
        BaseKerasDataGenerationTest(self, layer_weighting_type=BNLayerWeightingType.FIRST_LAYER_MULTIPLIER).run_test()

    def test_keras_bn_alignment_types(self):
        BaseKerasDataGenerationTest(self, bn_alignment_loss_type=BatchNormAlignemntLossType.L2_SQUARE).run_test()

    def test_keras_data_init_types(self):
        BaseKerasDataGenerationTest(self, data_init_type=DataInitType.Gaussian).run_test()

    def test_keras_image_granularity_types(self):
        BaseKerasDataGenerationTest(self, image_granularity=ImageGranularity.ImageWise).run_test()
        BaseKerasDataGenerationTest(self, image_granularity=ImageGranularity.BatchWise).run_test()
        BaseKerasDataGenerationTest(self, image_granularity=ImageGranularity.AllImages).run_test()

    def test_keras_image_pipeline_types(self):
        BaseKerasDataGenerationTest(self, image_pipeline_type=ImagePipelineType.IDENTITY).run_test()
        BaseKerasDataGenerationTest(self, image_pipeline_type=ImagePipelineType.SMOOTHING_AND_AUGMENTATION,
                                    extra_pixels=0).run_test()
        BaseKerasDataGenerationTest(self, image_pipeline_type=ImagePipelineType.SMOOTHING_AND_AUGMENTATION,
                                    image_clipping=False, extra_pixels=1).run_test()
        BaseKerasDataGenerationTest(self, image_pipeline_type=ImagePipelineType.SMOOTHING_AND_AUGMENTATION,
                                    image_clipping=True, extra_pixels=1).run_test()

    def test_keras_image_normalization_types(self):
        BaseKerasDataGenerationTest(self, image_normalization_type=ImageNormalizationType.KERAS_APPLICATIONS).run_test()
        BaseKerasDataGenerationTest(self, image_normalization_type=ImageNormalizationType.NO_NORMALIZATION).run_test()

    def test_keras_output_loss_types(self):
        BaseKerasDataGenerationTest(self, output_loss_type=OutputLossType.NONE).run_test()
        BaseKerasDataGenerationTest(self, output_loss_type=OutputLossType.NEGATIVE_MIN_MAX_DIFF, output_loss_multiplier=0.1).run_test()
        BaseKerasDataGenerationTest(self, output_loss_type=OutputLossType.INVERSE_MIN_MAX_DIFF, output_loss_multiplier=0.1).run_test()
        BaseKerasDataGenerationTest(self, output_loss_type=OutputLossType.REGULARIZED_MIN_MAX_DIFF, output_loss_multiplier=0.1).run_test()

    def test_keras_no_bn(self):
        with self.assertRaises(Exception) as e:
            BaseKerasDataGenerationTest(self, model=NoBNDataGenerationModel()).run_test()
        self.assertEqual('Data generation requires a model with at least one BatchNorm layer.', str(e.exception))
