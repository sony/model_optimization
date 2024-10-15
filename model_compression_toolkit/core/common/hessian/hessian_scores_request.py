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
from typing import Iterable, Sequence, Optional, TYPE_CHECKING
import dataclasses

from enum import Enum

if TYPE_CHECKING:    # pragma: no cover
    from model_compression_toolkit.core.common import BaseNode


class HessianMode(Enum):
    """
    Enum representing the mode for Hessian information computation.

    This determines whether the Hessian's approximation is computed w.r.t weights or w.r.t activations.
    Note: This is not the actual Hessian but an approximation.
    """
    WEIGHTS = 0         # Hessian approximation based on weights
    ACTIVATION = 1     # Hessian approximation based on activations


class HessianScoresGranularity(Enum):
    """
    Enum representing the granularity level for Hessian scores computation.

    This determines the number the Hessian scores is computed for some node.
    Note: This is not the actual Hessian but an approximation.
    """
    PER_ELEMENT = 0
    PER_OUTPUT_CHANNEL = 1
    PER_TENSOR = 2


@dataclasses.dataclass
class HessianScoresRequest:
    """
    Request configuration for the Hessian-approximation scores.

    This class defines the parameters for the scores based on the Hessian matrix approximation.
    It specifies the mode (weights/activations), granularity (element/channel/tensor), and the target node.

    Attributes:
        mode: Mode of Hessian-approximation score (w.r.t weights or activations).
        granularity: Granularity level for the approximation.
        target_nodes: The node objects in the float graph for which the Hessian's approximation scores is targeted.
        data_loader: Data loader to compute hessian approximations on. Should reflect the desired batch size for
            the computation. Can be None if all hessians for the request are expected to be pre-computed previously.
        n_samples: The number of samples to fetch hessian estimations for. If None, fetch hessians for a full pass
            of the data loader.
    """
    mode: HessianMode
    granularity: HessianScoresGranularity
    target_nodes: Sequence['BaseNode']
    data_loader: Optional[Iterable]
    n_samples: Optional[int]

    def __post_init__(self):
        if self.data_loader is None and self.n_samples is None:
            raise ValueError('Data loader and the number of samples cannot both be None.')

    def clone(self, **kwargs):
        """ Create a clone with optional overrides """
        return dataclasses.replace(self, **kwargs)
