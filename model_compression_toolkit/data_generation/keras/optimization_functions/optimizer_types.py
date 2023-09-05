from typing import Dict

import tensorflow_addons as tfa
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers.legacy import Optimizer
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow_addons.optimizers.rectified_adam import RectifiedAdam as RAdam

from model_compression_toolkit.data_generation.common.enums import OptimizerType

# Define a dictionary that maps optimizer types.
optimizers_dict: Dict[OptimizerType, Optimizer] = {
    OptimizerType.ADAM: Adam,
    OptimizerType.SGD: SGD,
    OptimizerType.RADAM: lambda lr: tfa.optimizers.Lookahead(RAdam(lr=lr), sync_period=6, slow_step_size=0.5),
}
