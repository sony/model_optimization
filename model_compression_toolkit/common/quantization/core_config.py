
from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.common.quantization.debug_config import DebugConfig
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import MixedPrecisionQuantizationConfigV2


class CoreConfig:
    """
    A class to hold the configurations classes of the MCT-core.
    """
    def __init__(self, n_iter: int = 500,
                 quantization_config: QuantizationConfig = QuantizationConfig(),
                 mixed_precision_config: MixedPrecisionQuantizationConfigV2 = None,
                 debug_config: DebugConfig = DebugConfig()
                 ):
        """

        Args:
            n_iter (int): Number of calibration iterations to run.
            quantization_config (QuantizationConfig): Config for quantization.
            mixed_precision_config (MixedPrecisionQuantizationConfigV2): Config for mixed precision quantization (optional,
            default=None).
            debug_config (DebugConfig): Config for debugging and editing the network quantization process.
        """
        self.n_iter = n_iter
        self.quantization_config = quantization_config
        self.mixed_precision_config = mixed_precision_config
        self.debug_config = debug_config

    @property
    def mixed_precision_enable(self):
        return self.mixed_precision_config is not None
