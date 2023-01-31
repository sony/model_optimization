# Training Methods for QAT

## Introduction

Several training methods may be applied by the user to train the QAT ready model
created by `keras_quantization_aware_training_init` method in [`keras/quantization_facade`](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/qat/keras/quantization_facade.py).
Each `TrainingMethod` (an enum defined in the [`qat_config`](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/qat/common/qat_config.py)) 
and [`QuantizationMethod`](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/common/target_platform/op_quantization_config.py)
selects a quantizer for weights and a quantizer for activations.

Currently, only the STE (straight through estimator) training method is implemented by the MCT.

## Make your own training method

Follow these steps in order to set the quantizers required by your training method:
- Add your `TrainingMethod` enum in [`qat_config`](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/qat/common/qat_config.py).
- Add your quantizers for weights and activation as explained in (https://github.com/sony/model_optimization/tree/main/model_compression_toolkit/quantizers_infrastructure/keras).
- Add your `TrainingMethod` and quantizers to `METHOD2WEIGHTQUANTIZER` and `METHOD2ACTQUANTIZER` in [`quantization_dispatcher_builder.py`](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/qat/keras/quantizer/quantization_dispatcher_builder.py)
according to your desired `QuantizationMethod`.  
- Set your `TrainingMethod` in the `QATConfig` and generate the QAT ready model for training. 

   
## Example: Adding a new training method

In this example we'll add a new quantization method, as described in this [`paper`](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710443.pdf).

First, we update the `TrainingMethod` enum in [`qat_config`](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/qat/common/qat_config.py)
```python
class TrainingMethod(Enum):
    """
    An enum for selecting a QAT training method

    STE - Standard straight-through estimator. Includes PowerOfTwo, symmetric & uniform quantizers
    HMQ - HMQ quantizer. Searches for PowerOfTwo thresholds and number of bits
    """
    STE = 0
    HMQ = 1
```

Then we implement the HMQ weight quantizer class for PoT thresholds: MyHMQWeightQuantizer

And update the quantizer selection dictionary `METHOD2WEIGHTQUANTIZER` in [`quantization_dispatcher_builder.py`](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/qat/keras/quantizer/quantization_dispatcher_builder.py)

```python
from my_quantizers import MyHMQWeightQuantizer

METHOD2WEIGHTQUANTIZER = {TrainingMethod.STE: {qi.QuantizationMethod.SYMMETRIC: STEWeightQuantizer,
                                               qi.QuantizationMethod.POWER_OF_TWO: STEWeightQuantizer,
                                               qi.QuantizationMethod.UNIFORM: STEUniformWeightQuantizer},
                          TrainingMethod.HMQ: {qi.QuantizationMethod.POWER_OF_TWO: MyHMQWeightQuantizer}
                          }
```

Finally, we're ready generate the model for quantization aware training
by calling `keras_quantization_aware_training_init` method in [`keras/quantization_facade`](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/qat/keras/quantization_facade.py)
with the following [`qat_config`](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/qat/common/qat_config.py):

```python
from model_compression_toolkit.qat.common.qat_config import QATConfig, TrainingMethod

qat_config = QATConfig(weight_training_method=TrainingMethod.HMQ)
```
