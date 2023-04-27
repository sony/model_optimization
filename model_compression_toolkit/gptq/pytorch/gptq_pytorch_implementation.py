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

from typing import Type

from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from model_compression_toolkit.gptq.common.gptq_framework_implementation import GPTQFrameworkImplemantation
from model_compression_toolkit.gptq.pytorch.gptq_training import PytorchGPTQTrainer


class GPTQPytorchImplemantation(GPTQFrameworkImplemantation, PytorchImplementation):

    def get_gptq_trainer_obj(self) -> Type[PytorchGPTQTrainer]:
        """
        Returns:  Pytorch object of GPTQTrainer
        """
        return PytorchGPTQTrainer