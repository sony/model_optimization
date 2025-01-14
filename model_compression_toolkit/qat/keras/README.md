# QAT Quantizers

## Introduction
All available training types for QAT are defined in the Enum [`TrainingMethod`](../../trainable_infrastructure/common/training_method.py).
A trainable quantizer can be Weights Quantizer or Activation Quantizer.
Any Activation Quantizer defined in [Activation Quantizers](../../trainable_infrastructure/keras/activation_quantizers) can be used for QAT.

## Make your own Keras trainable quantizers
In order to make your new quantizer you need to create your quantizer class, `MyTrainingQuantizer` and do as follows:
   - `MyTrainingQuantizer` should inherit from `BaseKerasQATWeightTrainableQuantizer` for weights quantizer or `BaseKerasActivationTrainableQuantizer` for activation quantizer
   - `MyTrainingQuantizer` should have `__init__` method that accepts `quantization_config` of type `TrainableQuantizerWeightsConfig` for weights quantizer or `TrainableQuantizerActivationConfig` for activation quantizer.
   - Implement `initialize_quantization` where you can define your parameters for the quantizer.
   - Implement `__call__` method to quantize the given inputs while training. This is your custom quantization itself. 
   - Implement `convert2inferable` method. This method exports your quantizer for inference (deployment). For doing that you need to choose one of the available Inferable Quantizers from the [MCT Quantizers](https://github.com/sony/mct_quantizers) package, according to the target when implementing `convert2inferable`, and set your learned quantization parameters there. 
   - Decorate `MyTrainingQuantizer` class with the `@mark_quantizer` decorator (provided by the [MCT Quantizers](https://github.com/sony/mct_quantizers) package) and choose the appropriate properties to set for your quantizer. The "identifier" property for the decorator should be of the type `TrainingMethod`  enum. See explanation about `@mark_quantizer` and how to use it under the [Keras Quantization Infrastructure](../../trainable_infrastructure/keras/README.md).
   

   
## Example: Symmetric Weights Quantizer
To create custom `MyWeightsTrainingQuantizer` which is a symmetric weights training quantizer you need to set
`QuantizationTarget.Weights` as target and `QuantizationMethod.SYMMETRIC` as method.
Assume that the quantizer has a new training method called `MyTrainig` which is defined in the `TrainingMethod` Enum.

```python
import tensorflow as tf

from model_compression_toolkit.trainable_infrastructure import TrainingMethod, TrainableQuantizerWeightsConfig
from mct_quantizers import mark_quantizer, QuantizationMethod, QuantizationTarget
from mct_quantizers.keras.quantizers import WeightsUniformInferableQuantizer
from model_compression_toolkit.qat.keras.quantizer.base_keras_qat_weight_quantizer import BaseKerasQATWeightTrainableQuantizer

NEW_PARAM = "new_param_name"

@mark_quantizer(quantization_target=QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.SYMMETRIC],
                identifier=TrainingMethod.MyTraining)
class MyWeightsTrainingQuantizer(BaseKerasQATWeightTrainableQuantizer):
    def __init__(self, quantization_config: TrainableQuantizerWeightsConfig):
        super(MyWeightsTrainingQuantizer, self).__init__(quantization_config)
        # Define your new params here:
        self.new_param = ...

    def initialize_quantization(self, tensor_shape, name, layer):
        # Creating new params for quantizer
        new_param = layer.add_weight(NEW_PARAM,
                                     shape=len(tensor_shape),
                                     initializer=tf.keras.initializers.Constant(1.0),
                                     trainable=True)
        new_param.assign(self.new_param)
        # Save the quantizer parameters for later calculations
        self.quantizer_parameters = {NEW_PARAM: new_param}
        return self.quantizer_parameters

    def __call__(self, inputs):
        # Your quantization logic here
        new_param = self.quantizer_parameters[NEW_PARAM]
        # Custom quantization function you need to implement
        quantized_inputs = custom_quantize(inputs, new_param)
        return quantized_inputs

    def convert2inferable(self):
        return WeightsUniformInferableQuantizer(...)

```

## Example: Symmetric Activations Quantizer
To create custom `MyActivationsTrainingQuantizer` which is a symmetric activations training quantizer you need to set `QuantizationTarget.Activation` as target and `QuantizationMethod.SYMMETRIC` as method.
Assume that the quantizer has a new training method called `MyTrainig` which is defined in the `TrainingMethod` Enum.

```python
import tensorflow as tf

from mct_quantizers.keras.quantizers import ActivationUniformInferableQuantizer
from model_compression_toolkit.trainable_infrastructure import TrainingMethod, TrainableQuantizerActivationConfig
from mct_quantizers import QuantizationMethod
from mct_quantizers import mark_quantizer, QuantizationTarget
from model_compression_toolkit.qat.keras.quantizer.base_keras_qat_weight_quantizer import BaseKerasQATWeightTrainableQuantizer

NEW_PARAM = "new_param_name"

@mark_quantizer(quantization_target=QuantizationTarget.Activation,
                quantization_method=[QuantizationMethod.SYMMETRIC],
                identifier=TrainingMethod.TrainingMethod)
class MyActivationsTrainingQuantizer(BaseKerasQATWeightTrainableQuantizer):
    def __init__(self, quantization_config: TrainableQuantizerActivationConfig):
        super(MyActivationsTrainingQuantizer, self).__init__(quantization_config)
        # Define your new params here:
        self.new_param = ...

    def initialize_quantization(self, tensor_shape, name, layer):
        # Creating new params for quantizer
        new_param = layer.add_weight(NEW_PARAM,
                                     shape=len(tensor_shape),
                                     initializer=tf.keras.initializers.Constant(1.0),
                                     trainable=True)
        new_param.assign(self.new_param)
        # Save the quantizer parameters for later calculations
        self.quantizer_parameters = {NEW_PARAM: new_param}
        return self.quantizer_parameters

    def __call__(self, inputs):
        # Your quantization logic here
        new_param = self.quantizer_parameters[NEW_PARAM]
        # Custom quantization function you need to implement
        quantized_inputs = custom_quantize(inputs, new_param)
        return quantized_inputs

    def convert2inferable(self):
        return ActivationUniformInferableQuantizer(...)
```
