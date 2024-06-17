#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
from model_compression_toolkit.core.pytorch.utils import get_working_device

from model_compression_toolkit.ptq.pytorch.quantization_facade import DEFAULT_PYTORCH_TPC
from model_compression_toolkit.xquant.common.framework_report_utils import FrameworkReportUtils
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from model_compression_toolkit.xquant.common.model_folding_utils import ModelFoldingUtils
from model_compression_toolkit.xquant.common.similarity_calculator import SimilarityCalculator
from model_compression_toolkit.xquant.pytorch.dataset_utils import PytorchDatasetUtils
from model_compression_toolkit.xquant.pytorch.model_analyzer import PytorchModelAnalyzer
from model_compression_toolkit.xquant.pytorch.similarity_functions import PytorchSimilarityFunctions
from model_compression_toolkit.xquant.pytorch.tensorboard_utils import PytorchTensorboardUtils


class PytorchReportUtils(FrameworkReportUtils):
    """
    Class with various utility components required for generating the report for a Pytorch model.
    """
    def __init__(self, report_dir: str):
        """
        Args:
            report_dir: Logging dir path.
        """
        fw_info = DEFAULT_PYTORCH_INFO
        fw_impl = PytorchImplementation()

        dataset_utils = PytorchDatasetUtils()
        model_folding = ModelFoldingUtils(fw_info=fw_info,
                                          fw_impl=fw_impl,
                                          fw_default_tpc=DEFAULT_PYTORCH_TPC)

        similarity_calculator = SimilarityCalculator(dataset_utils=dataset_utils,
                                                     model_folding=model_folding,
                                                     similarity_functions=PytorchSimilarityFunctions(),
                                                     model_analyzer_utils=PytorchModelAnalyzer(),
                                                     device=get_working_device())

        tb_utils = PytorchTensorboardUtils(report_dir=report_dir,
                                           fw_impl=fw_impl,
                                           fw_info=fw_info)

        super().__init__(fw_info=fw_info,
                         fw_impl=fw_impl,
                         tb_utils=tb_utils,
                         dataset_utils=dataset_utils,
                         similarity_calculator=similarity_calculator,
                         model_folding_utils=model_folding)
