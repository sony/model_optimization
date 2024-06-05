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
import logging

from typing import Any, Dict, Callable, Tuple

from xquant.common.constants import CS_METRIC_NAME, SQNR_METRIC_NAME, MSE_METRIC_NAME

DEFAULT_METRICS_NAMES = [CS_METRIC_NAME, MSE_METRIC_NAME, SQNR_METRIC_NAME]


class SimilarityFunctions:

    @staticmethod
    def compute_mse(f_pred: Any, q_pred: Any) -> float:
        raise NotImplemented

    @staticmethod
    def compute_cs(f_pred: Any, q_pred: Any) -> float:
        raise NotImplemented

    @staticmethod
    def compute_sqnr(f_pred: Any, q_pred: Any) -> float:
        raise NotImplemented

    def get_default_metrics(self) -> Dict[str, Callable]:
        return {
            MSE_METRIC_NAME: self.compute_mse,
            CS_METRIC_NAME: self.compute_cs,
            SQNR_METRIC_NAME: self.compute_sqnr
        }

