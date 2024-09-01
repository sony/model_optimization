# Training Methods for QAT

## Introduction

Several training methods may be applied by the user to train the QAT ready model
created by `pytorch_quantization_aware_training_init` method in [`pytorch/quantization_facade`](../quantization_facade.py).
Each [`TrainingMethod`](../../../trainable_infrastructure/common/training_method.py) 
and [`QuantizationMethod`](../../../target_platform_capabilities/target_platform/op_quantization_config.py)
selects a quantizer for weights and a quantizer for activations.

## Make your own training method

Follow these steps in order to set the quantizers required by your training method:
- Add your training method to the `TrainingMethod` enum.
- Add your quantizers for weights and activation as explained in [quantizer readme](../../../trainable_infrastructure/pytorch).
- Import your quantizer package in the quantizer [`__init.py__`](./__init__.py) file.
- Set your `TrainingMethod` in the `QATConfig` and generate the QAT ready model for training. 

   
## Example: Adding a new training method

In this example we'll add a new quantization method, called MTM (my training method).

First, we update the `TrainingMethod`(../../../trainable_infrastructure/common/training_method.py)
```python
class TrainingMethod(Enum):
    """
    An enum for selecting a QAT training method

    STE - Standard straight-through estimator. Includes PowerOfTwo, symmetric & uniform quantizers
    MTM - MyTrainingMethod.
    """
    STE = 0
    MTM = 1
```

Then we implement a weight quantizer class that implements the desired training scheme: MTMWeightQuantizer,
under a new package in `qat/pytorch/quantizer/mtm_quantizer/mtm.py`, and import it in the quantizer `__init__.py` file.

```python
import model_compression_toolkit.qat.pytorch.quantizer.mtm_quantizer.mtm
```

Finally, we're ready to generate the model for quantization aware training
by calling `pytorch_quantization_aware_training_init` method in [`pytorch/quantization_facade`](../quantization_facade.py)
with the following [`qat_config`](../../common/qat_config.py):

```python
from model_compression_toolkit.qat.common.qat_config import QATConfig
from model_compression_toolkit.trainable_infrastructure import TrainingMethod

qat_config = QATConfig(weight_training_method=TrainingMethod.MTM)
```
