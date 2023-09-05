from typing import Dict

from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers.legacy import Optimizer
from tensorflow.keras.optimizers.legacy import SGD

from model_compression_toolkit.data_generation.common.enums import OptimizerType

# Define a dictionary that maps optimizer types.
optimizers_dict: Dict[OptimizerType, Optimizer] = {
    OptimizerType.ADAM: Adam,
    OptimizerType.SGD: SGD,
}
