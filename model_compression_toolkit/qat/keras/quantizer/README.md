## Introduction

The `BaseKerasTrainableQuantizer` class is a base class for Keras quantizers that are used for training only. It is a subclass of `BaseTrainableQuantizer` and is designed to be used as a base class for creating custom quantization methods for Keras models.

## Installation

To use the `BaseKerasTrainableQuantizer` class, you will need to have Tensorflow and tensorflow_model_optimization installed, otherwise an exception will be raised.

Once you have Tensorflow installed, you can use the `BaseKerasTrainableQuantizer` class by importing it from the package where it is located.

## Usage

The `BaseKerasTrainableQuantizer` class takes single argument during initialization:

- `quantization_config`: quantization configuration object which can be `NodeWeightsQuantizationConfig` or `NodeActivationQuantizationConfig` for weights or activations respectively.

Once you have instantiated the class, you can use the `__call__` method to quantize the given inputs while training using the quantizer parameters. 
The method returns the quantized tensor. You must implement the abstract method `__call__` in your subclass which inherits `BaseKerasTrainableQuantizer`.

In order to export the quantizer for deployment you also must to implement `convert2inferable` method. For that you need to create your Inferable Quantizer (see [InferableQuantizers](https://github.com/sony/model_optimization/tree/main/model_compression_toolkit/quantizers_infrastructure/keras/inferable_quantizers/README.md))

Example for custom symmetric weights training quantizer.
First, add enum to `TrainingMethod` in `qat/common/qat_config.py` your custom training method and set your custom quantizer to `quantization_dispatcher_builder.py` dictionary.
```python
class TrainingMethod(Enum):
    STE = 0
    MyTrainingMethod = 1
```
```python
METHOD2WEIGHTQUANTIZER = { ...,
                           TrainingMethod.MyTrainingMethod: {qi.QuantizationMethod.SYMMETRIC: MyWeightsTrainingQuantizer}}
```
then, create your custom quantizer:
```python
from model_compression_toolkit import quantizers_infrastructure as qi
class MyWeightsTrainingQuantizer(BaseKerasTrainableQuantizer):
    def __init__(self, quantization_config: NodeWeightsQuantizationConfig):
        super(MyWeightsTrainingQuantizer, self).__init__(quantization_config,
                                                 qi.QuantizationTarget.Weights,
                                                 [qi.QuantizationMethod.SYMMETRIC])
    def __call__(self, inputs):
        quantized = inputs.round()
        return quantized

    def convert2inferable(self):
        return MyWeightsInferableQuantizer()
```

Example for custom symmetric activations training quantizer:
```python
METHOD2ACTQUANTIZER = { ...,
                           TrainingMethod.MyTrainingMethod: {qi.QuantizationMethod.SYMMETRIC: MyActivationsTrainingQuantizer}}
```
```python
from model_compression_toolkit import quantizers_infrastructure as qi
class MyActivationsTrainingQuantizer(BaseKerasTrainableQuantizer):
    def __init__(self, quantization_config: NodeActivationQuantizationConfig):
        super(MyActivationsTrainingQuantizer, self).__init__(quantization_config,
                                                 qi.QuantizationTarget.Activation,
                                                 [qi.QuantizationMethod.SYMMETRIC])
    def __call__(self, inputs):
        quantized = inputs.round()
        return quantized

    def convert2inferable(self):
        return MyActivationsInferableQuantizer()
```

##Note

Keep in mind that `BaseKerasTrainableQuantizer` is an abstract class, it should not be instantiated directly. You should create a new class that inherits from it and implements the required methods.

If you have any questions or issues using the BaseKerasTrainableQuantizer class, please open an issue on the GitHub repository or reach out to the maintainers for assistance.