import unittest

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from model_compression_toolkit.data_generation.common.enums import SchedularType, BatchNormAlignemntLossType, \
    DataInitType, LayerWeightingType, ImageGranularity, ImagePipelineType, ImageNormalizationType, OutputLossType
from tests.data_generation_tests.pytorch.base_pytorch_data_generation_test import BasePytorchDataGenerationTest


class PytorchDataGenerationTestRunner(unittest.TestCase):
    def test_pytorch_schedular_types(self):
        BasePytorchDataGenerationTest(self, scheduler=StepLR, scheduler_type=SchedularType.STEP).run_test()
        BasePytorchDataGenerationTest(self, scheduler=ReduceLROnPlateau, scheduler_type=SchedularType.REDUCE_ON_PLATEAU).run_test()

    def test_pytorch_layer_weighting_types(self):
        BasePytorchDataGenerationTest(self, layer_weighting_type=LayerWeightingType.AVERAGE).run_test()

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
        BasePytorchDataGenerationTest(self, image_pipeline_type=ImagePipelineType.CROP_FLIP, image_padding=0).run_test()
        BasePytorchDataGenerationTest(self, image_pipeline_type=ImagePipelineType.CROP_FLIP, image_padding=1).run_test()

    def test_pytorch_image_normalization_types(self):
        BasePytorchDataGenerationTest(self, image_normalization_type=ImageNormalizationType.TORCHVISION).run_test()
        BasePytorchDataGenerationTest(self, image_normalization_type=ImageNormalizationType.NO_NORMALIZATION).run_test()

    def test_pytorch_output_loss_types(self):
        BasePytorchDataGenerationTest(self, output_loss_type=OutputLossType.NONE).run_test()
        BasePytorchDataGenerationTest(self, output_loss_type=OutputLossType.MIN_MAX_DIFF).run_test()

if __name__ == '__main__':
    unittest.main()