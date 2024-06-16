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


from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.xquant.common.framework_report_utils import FrameworkReportUtils
from model_compression_toolkit.ptq.keras.quantization_facade import DEFAULT_KERAS_TPC
from model_compression_toolkit.xquant.common.model_folding_utils import ModelFoldingUtils
from model_compression_toolkit.xquant.common.similarity_calculator import SimilarityCalculator
from model_compression_toolkit.xquant.keras.dataset_utils import KerasDatasetUtils
from model_compression_toolkit.xquant.keras.model_analyzer import KerasModelAnalyzer

from model_compression_toolkit.xquant.keras.similarity_functions import KerasSimilarityFunctions
from model_compression_toolkit.xquant.keras.tensorboard_utils import KerasTensorboardUtils


class KerasReportUtils(FrameworkReportUtils):
    """
    Class with various utility components required for generating the report for a Keras model.
    """
    def __init__(self, report_dir: str):
        """
        Args:
            report_dir: Logging dir path.
        """
        fw_info = DEFAULT_KERAS_INFO
        fw_impl = KerasImplementation()

        dataset_utils = KerasDatasetUtils()
        model_folding = ModelFoldingUtils(fw_info=fw_info,
                                          fw_impl=fw_impl,
                                          fw_default_tpc=DEFAULT_KERAS_TPC)

        similarity_calculator = SimilarityCalculator(dataset_utils=dataset_utils,
                                                     model_folding=model_folding,
                                                     similarity_functions=KerasSimilarityFunctions(),
                                                     model_analyzer_utils=KerasModelAnalyzer())

        tb_utils = KerasTensorboardUtils(report_dir=report_dir,
                                         fw_impl=fw_impl,
                                         fw_info=fw_info)
        super().__init__(fw_info,
                         fw_impl,
                         similarity_calculator,
                         dataset_utils,
                         model_folding,
                         tb_utils)
