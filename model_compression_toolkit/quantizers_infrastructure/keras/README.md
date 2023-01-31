## Introduction

[`BaseKerasTrainableQuantizer`](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/quantizers_infrastructure/keras/base_Keras_quantizer.py) is an interface that enables easy quantizers development and training. 
Using this base class makes it simple to implement new quantizers for training and inference for weights or activations.

## Make your own Keras trainable quantizers
Trainable quantizer can be Weights Quantizer or Activation Quantizer.
In order to make your new quantizer you need to create your quantizer class, `MyTrainingQuantizer` and do as follows:
   - `MyTrainingQuantizer` should inherit from [`BaseKerasTrainableQuantizer`](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/quantizers_infrastructure/keras/base_Keras_quantizer.py).
   - `MyTrainingQuantizer` should have [`init`](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/quantizers_infrastructure/common/base_trainable_quantizer.py) function that gets `quantization_config` which is [`NodeWeightsQuantizationConfig`](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/common/quantization/node_quantization_config.py#L228) if you choose to implement weights quantizer or [`NodeActivationQuantizationConfig`](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/common/quantization/node_quantization_config.py#L63) if you choose activation quantizer. This initialization function also defines:
     - [`QuantizationTarget`](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/quantizers_infrastructure/common/base_inferable_quantizer.py): Enum indicating weights or activations.
     - [`QuantizationMethod`](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/common/target_platform/op_quantization_config.py): List of quantization methods (Uniform, Symmetric, etc.)
   - Implement [`initialize_quantization`](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/quantizers_infrastructure/common/base_trainable_quantizer.py) where you can define your parameters for the quantizer.
   - Implement [`__call__`](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/quantizers_infrastructure/common/base_trainable_quantizer.py) method to quantize the given inputs while training. This is your custom quantization itself. 
   - Implement [`convert2inferable`](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/quantizers_infrastructure/common/base_trainable_quantizer.py) method. This method exports your quantizer for inference (deployment). For doing that you need to choose one of our Inferable Quantizers ([Inferable Quantizers](https://github.com/sony/model_optimization/tree/main/model_compression_toolkit/quantizers_infrastructure/keras/inferable_quantizers)) according to target when implement `convert2inferable`, and set your learned quantization parameters there. 
   
## Example: Symmetric Weights Quantizer
To create custom `MyWeightsTrainingQuantizer` which is a symmetric weights training quantizer you need to set
`qi.QuantizationTarget.Weights` as target and `qi.QuantizationMethod.SYMMETRIC` as method.
```python
import tensorflow as tf
from model_compression_toolkit import quantizers_infrastructure as qi
NEW_PARAM = "new_param_name"
class MyWeightsTrainingQuantizer(BaseKerasTrainableQuantizer):
    def __init__(self, quantization_config: NodeWeightsQuantizationConfig):
        super(MyWeightsTrainingQuantizer, self).__init__(quantization_config,
                                                 qi.QuantizationTarget.Weights,
                                                 [qi.QuantizationMethod.SYMMETRIC])
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
        return qi.WeightsUniformInferableQuantizer(...)
```

## Example: Symmetric Activations Quantizer
To create custom `MyActivationsTrainingQuantizer` which is a symmetric activations training quantizer you need to set `qi.QuantizationTarget.Activation` as target and `qi.QuantizationMethod.SYMMETRIC` as method.
```python
import tensorflow as tf
NEW_PARAM = "new_param_name"
from model_compression_toolkit import quantizers_infrastructure as qi
class MyActivationsTrainingQuantizer(BaseKerasTrainableQuantizer):
    def __init__(self, quantization_config: NodeActivationQuantizationConfig):
        super(MyActivationsTrainingQuantizer, self).__init__(quantization_config,
                                                 qi.QuantizationTarget.Activation,
                                                 [qi.QuantizationMethod.SYMMETRIC])
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
        return qi.ActivationUniformInferableQuantizer(...)
```

## Fully implementation quantizers
For fully reference, check our QAT quantizers here:
[QAT Quantizers](https://github.com/sony/model_optimization/tree/main/model_compression_toolkit/qat/keras/quantizer/ste_rounding)