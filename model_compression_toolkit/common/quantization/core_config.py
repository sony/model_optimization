
from typing import List
from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import MixedPrecisionQuantizationConfigV2
from model_compression_toolkit.common.network_editors.edit_network import EditRule


class CoreConfig:
    def __init__(self, n_iter: int = 500,
                 quantization_config: QuantizationConfig = QuantizationConfig(),
                 mixed_precision_config: MixedPrecisionQuantizationConfigV2 = None,
                 network_editor: List[EditRule] = []):
        """

        Args:

            kpi: Model maximal memory size (in bytes) for mixed precision optimization for weights & activations quantization
            quantization_config: quantization config
            mixed_precision_config: mixed precision config (optional)
        """
        self.n_iter = n_iter
        self.quantization_config = quantization_config
        self.mixed_precision_config = mixed_precision_config
        self.network_editor = network_editor

    @property
    def mixed_precision_enable(self):
        return self.mixed_precision_config is not None
