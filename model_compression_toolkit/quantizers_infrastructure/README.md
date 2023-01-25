# Quantizers Infrastructure (QI)

Quantizers infrastructure is a library that provides quantization modules for hardware-oriented model optimization tools.

The quantization modules (quantizers) can be used for emulating inference-time qunatization (inferable-quantizers) and for optimization of model quantization during training (trainable-quantizers).

## Library structure

### Quantizer

Quantizer instance implements the quantization function: conversion of floating point values to quantized values represented in floating point (fake-quantization).
Quantizer instance should be defined per layer, separately for weights quantization and activations quantization.

The quntizers are divided into two main types:

`BaseInferableQuantizer` - Type of quantizers that use for emulating inference-time quantization   

`BaseTrainableQuantizer` - Type of quantizers that use for quantization optimization during training. This type of quntizers contains learnable parameters

Usage example:

```python
from model_compression_toolkit import quantizers_infrastructure as qi

# generate signed 8-bits symmetric quantizer for numbers in the interval [-2,2) 
w_quantizer = qi.WeightsSymmetricInferableQuantizer(num_bits=8,
                                                    threshold=2,
                                                    signed=True,
                                                    per_channel=True)
```
### Dispatcher

`NodeQuantizationDispatcher` instance contains a list of relevant quantizers for node (layer) in the model:
weights quantizer per attribute (kernel, bias, ...)
activation quantizer per layer's output

Usage example:

```python
# generate node dispatcher with the above weights quantizer 
dispatcher = qi.KerasNodeQuantizationDispatcher(w_quantizer)
```
### Quantization Wrapper

Quantization wrapper instance is used as a layer's replacement which emulates the layer's operation only with quantization.
The quantization wrapper receives the layer to wrap, and a dispatcher for mapping the relevant quantizer per attribute. 

`KerasQuantizationWrapper` - used to wrap Keras layer

`PytorchQuantizationWrapper` - used to wrap torch nn module

Usage example:

```python
from tensorflow import keras

# generate example of Keras model
inputs = keras.layers.Input(shape=(32,32,3))
x = keras.layers.Conv2D(6, 7, use_bias=False)(inputs)
model = keras.Model(inputs=inputs, outputs=x)

# create quantization wrapper from the convolution layer according to the dispatcher
qi.KerasQuantizationWrapper(layer=model.layers[1], dispatcher)
```

### Quantization configuration

Quantizer configuration or quantizer parameters can be stored in `BaseQuantizerConfig` object. 

For trainable quantizer we use:

`TrainableQuantizerWeightsConfig` - contain the configuration of weight quantization for trainable quantizer.

`TrainableQuantizerActivationConfig` - contain the configuration of activation quantization for trainable quantizer

For inferable quantizers we don't use the `BaseQuantizerConfig` object, instead we use explicit parameters per quantizer  

Usage example:
```python
# generate example configurations for trainable quantizer
q_config = qi.TrainableQuantizerWeightsConfig(weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
                                              weights_n_bits=9,
                                              weights_quantization_params={},
                                              enable_weights_quantization=True,
                                              weights_channels_axis=-1,
                                              weights_per_channel_threshold=True,
                                              min_threshold=0)


```



