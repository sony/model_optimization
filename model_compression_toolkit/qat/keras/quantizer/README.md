# Training Methods for QAT

## Introduction

Several training methods may be applied by the user to train the QAT ready model
created by `keras_quantization_aware_training_init` method in [`keras/quantization_facade`](../quantization_facade.py).
Each `TrainingMethod` (an enum defined in the [`qat_config`](../../common/qat_config.py)) 
and [`QuantizationMethod`](../../../core/common/target_platform/op_quantization_config.py)
selects a quantizer for weights and a quantizer for activations.

Currently, only the STE (straight through estimator) training method is implemented by the MCT.

## Make your own training method

Follow these steps in order to set the quantizers required by your training method:
- Add your `TrainingMethod` enum in [`qat_config`](../../common/qat_config.py).
- Add your quantizers for weights and activation as explained in [quantizer readme](../../quantizers_infrastructure/keras).
- Add your `TrainingMethod` and quantizers to `METHOD2WEIGHTQUANTIZER` and `METHOD2ACTQUANTIZER` in [`quantization_dispatcher_builder.py`](../quantizer/quantization_dispatcher_builder.py)
according to your desired `QuantizationMethod`.  
- Set your `TrainingMethod` in the `QATConfig` and generate the QAT ready model for training. 

   
## Example: Adding a new training method

In this example we'll add a new quantization method, called MTM (my training method).

First, we update the `TrainingMethod` enum in [`qat_config`](../../common/qat_config.py)
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

Then we implement a weight quantizer class that implements the desired training scheme: MTMWeightQuantizer

And update the quantizer selection dictionary `METHOD2WEIGHTQUANTIZER` in [`quantization_dispatcher_builder.py`](../quantizer/quantization_dispatcher_builder.py)

```python
from my_quantizers import MTMWeightQuantizer

METHOD2WEIGHTQUANTIZER = {TrainingMethod.STE: {qi.QuantizationMethod.SYMMETRIC: STEWeightQuantizer,
                                               qi.QuantizationMethod.POWER_OF_TWO: STEWeightQuantizer,
                                               qi.QuantizationMethod.UNIFORM: STEUniformWeightQuantizer},
                          TrainingMethod.MTM: {qi.QuantizationMethod.POWER_OF_TWO: MTMWeightQuantizer}
                          }
```

Finally, we're ready to generate the model for quantization aware training
by calling `keras_quantization_aware_training_init` method in [`keras/quantization_facade`](../quantization_facade.py)
with the following [`qat_config`](../../common/qat_config.py):

```python
from model_compression_toolkit.qat.common.qat_config import QATConfig, TrainingMethod

qat_config = QATConfig(weight_training_method=TrainingMethod.MTM)
```
