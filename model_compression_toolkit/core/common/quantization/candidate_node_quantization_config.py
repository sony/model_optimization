# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
import copy
from dataclasses import dataclass, InitVar
from typing import Callable, List, Optional

from model_compression_toolkit.core.common.quantization.node_quantization_config import BaseNodeQuantizationConfig, \
    NodeWeightsQuantizationConfig, NodeActivationQuantizationConfig, ActivationQuantizationMode


@dataclass(eq=True)
class CandidateNodeQuantizationConfig(BaseNodeQuantizationConfig):
    """
    Candidate quantization configuration for a node.
    """
    activation_quantization_cfg: NodeActivationQuantizationConfig
    # TODO irena: None is passed in several places, need to check if it's handled properly or it's only passed in cases
    #  that do not affect anything (my guess is it's the second).
    #  I think in general it makes more sense to set it to None when there are no weights, and maybe when all weights
    #  are unquantized, and handle it properly everywhere.
    weights_quantization_cfg: Optional[NodeWeightsQuantizationConfig]


# TODO irena: currently all code still looks at candidates_quantization_cfg as previously, so this is just an initial
#  implementation. For now base config is completely separated from candidates (base config must be equal to one of the
#  candidates, but we create a separate copy), and updating in place is allowed. Also we require quantization mode to
#  be identical between all configs.
@dataclass
class NodeQuantizationConfig:
    # quantization config for single precision
    base_quantization_cfg: CandidateNodeQuantizationConfig
    # quantization candidate configs for mixed precision
    candidates_quantization_cfg: List[CandidateNodeQuantizationConfig]

    validate: InitVar[bool] = True

    def update_all(self, update_fn: Callable[[CandidateNodeQuantizationConfig], None]):
        """
        Apply update function on the base config and all candidates configs.

        Args:
            update_fn: function to apply.
        """
        if self.base_quantization_cfg:
            update_fn(self.base_quantization_cfg)
        for cfg in self.candidates_quantization_cfg:
            update_fn(cfg)

    def update_activation_quantization_mode(self, mode: ActivationQuantizationMode):
        """
        Update activation quantization mode for the base config and all candidates configs.

        Args:
            mode: quantization mode.
        """
        def fn(c):
            c.activation_quantization_cfg.quant_mode = mode

        self.update_all(fn)

    def disable_weights_quantization(self):
        """
        Disable all weights quantization for the base config and all candidates configs.
        """
        self.update_all(lambda c: c.weights_quantization_cfg.disable_all_weights_quantization())

    def get_activation_quant_mode(self) -> ActivationQuantizationMode:
        """
        Retrieve activation quantization mode.

        Returns:
            Activation quantization mode.

        Raises:
            ValueError if not all candidates contain the same mode.
        """
        self._validate_consistent_activation_quant_mode()
        return self.base_quantization_cfg.activation_quantization_cfg.quant_mode

    def __post_init__(self, validate=True):
        if validate:
            if not any(self.base_quantization_cfg == qc for qc in self.candidates_quantization_cfg):
                raise ValueError('Candidates should contain the base config.')
            self._validate_consistent_activation_quant_mode()
            self._validate_consistent_weights_quant_mode()
        # TODO irena
        # for now make sure they are separate objects so that one doesnt inadvertently modify the other
        if any(self.base_quantization_cfg is qc for qc in self.candidates_quantization_cfg):
            self.base_quantization_cfg = copy.deepcopy(self.base_quantization_cfg)

    def _validate_consistent_activation_quant_mode(self):
        """
        Validate that base config and all candidates configs contain identical activation quantization mode.

        Raises:
            ValueError if activation quantization mode is not consistent.
        """
        activation_quant_mode = self.base_quantization_cfg.activation_quantization_cfg.quant_mode
        if any(qc.activation_quantization_cfg.quant_mode != activation_quant_mode
               for qc in self.candidates_quantization_cfg):
            raise ValueError('Quantization candidates with different quantization modes are not currently supported.')

    def _validate_consistent_weights_quant_mode(self):
        """
        Validate that base config and all candidates configs contain identical weights quantization mode per attribute,
        i.e. quantization for each attribute should either be enabled in all configs, or disabled in all configs.

        Raises:
            ValueError if weights quantization is not consistent.
        """
        def get_weights_mode(qc):
            # in graph fuser weights_quantization_cfg is set to None
            if qc.weights_quantization_cfg is None:
                return None
            return {attr: attr_cfg.enable_weights_quantization for attr, attr_cfg
                    in qc.weights_quantization_cfg.get_all_weight_attrs_configs().items()}
        if any(get_weights_mode(self.base_quantization_cfg) != get_weights_mode(qc)
               for qc in self.candidates_quantization_cfg):
            raise ValueError('Quantization candidates with different quantization modes are not currently supported.')
