# Model Compression Toolkit (MCT)

Model Compression Toolkit (MCT) is an open-source project for neural network model optimization under efficient, constrained hardware.

This project provides researchers, developers, and engineers tools for optimizing and deploying state-of-the-art neural networks on efficient hardware.

Specifically, this project aims to apply quantization to compress neural networks.

<img src="docsrc/images/mct_block_diagram.svg" width="10000">

MCT is developed by researchers and engineers working at Sony Semiconductor Israel.



## Table of Contents

- [Supported features](#supported-features)
- [Getting Started](#getting-started)
- [Results](#results)
- [Contributions](#contributions)
- [License](#license)

## Supported Features

MCT supports different quantization methods:
* Post training quantization (PTQ): [Keras API](https://sony.github.io/model_optimization/docs/api/experimental_api_docs/methods/keras_post_training_quantization_experimental.html#ug-keras-post-training-quantization-experimental), [PyTorch API](https://sony.github.io/model_optimization/docs/api/experimental_api_docs/methods/pytorch_post_training_quantization_experimental.html#ug-pytorch-post-training-quantization-experimental)
* Gradient-based post training quantization (GPTQ): [Keras API](https://sony.github.io/model_optimization/docs/api/experimental_api_docs/methods/keras_gradient_post_training_quantization_experimental.html#ug-keras-gradient-post-training-quantization-experimental), [PyTorch API](https://sony.github.io/model_optimization/docs/api/experimental_api_docs/methods/pytorch_gradient_post_training_quantization_experimental.html#ug-pytorch-gradient-post-training-quantization-experimental)
* Quantization aware training (QAT)[*](#experimental-features)


| Quantization Method | Complexity                                    | Computational Cost          |
|---------------------|-----------------------------------------------|-----------------------------|
| PTQ                 | Low                                           | Low (order of minutes)      |
| GPTQ                | Mild (parameters fine-tuning using gradients) | Mild (order of 2-3 hours)   |
| QAT                 | High                                          | High (order of 12-36 hours) |


In addition, MCT supports different quantization schemes for quantizing weights and activations:
* Power-Of-Two (hardware-friendly quantization [1])
* Symmetric
* Uniform

Core features:
* <ins>Graph optimizations:</ins> Transforming the model to an equivalent (yet, more efficient) model (for example, batch-normalization layer folding to its preceding linear layer).
* <ins>Quantization parameter search:</ins> Different methods can be used to minimize the expected added quantization-noise during thresholds search (by default, we use Mean-Square-Errorm but other metrics can be used such as No-Clipping, Mean-Average-Error, and more).
* <ins>Advanced quantization algorithms:</ins> To prevent a performance degradation some algorithms are applied such as: 
  * <ins>Shift negative correction:</ins> Symmetric activation quantization can hurt the model's performance when some layers output both negative and positive activations, but their range is asymmetric. For more details please visit [1].
  * <ins>Outliers filtering:</ins> Computing z-score for activation statistics to detect and remove outliers.
* <ins>Clustering:</ins> Using non-uniform quantization grid to quantize the weights and activations to match their distributions.[*](#experimental-features)
* <ins>Mixed-precision search:</ins> Assigning quantization bit-width per layer (for weights/activations), based on the layer's sensitivity to different bit-widths.
* <ins>Visualization:</ins> You can use TensorBoard to observe useful information for troubleshooting the quantized model's performance (for example, the model in different phases of the quantization, collected statistics, similarity between layers of the float and quantized model and bit-width configuration for mixed-precision quantization). For more details, please read the [visualization documentation](https://sony.github.io/model_optimization/docs/guidelines/visualization.html).   


#### Experimental features 

Some features are experimental and subject to future changes. 
 
For more details, we highly recommend visiting our project website where experimental features are mentioned as experimental.


## Getting Started

This section provides a quick starting guide. We begin with installation via source code or pip server. Then, we provide a short usage example.

### Installation
See the MCT install guide for the pip package, and build from the source.


#### From Source
```
git clone https://github.com/sony/model_optimization.git
python setup.py install
```
#### From PyPi - latest stable release
```
pip install model-compression-toolkit
```

A nightly package is also available (unstable):
```
pip install mct-nightly
```

### Requierments

To run MCT, one of the supported frameworks, Tenosflow/Pytorch, needs to be installed.

For using with Tensorflow please install the packages: 
[tensorflow](https://www.tensorflow.org/install), 
[tensorflow-model-optimization](https://www.tensorflow.org/model_optimization/guide/install)

For using with PyTorch please install the packages: 
[torch](https://pytorch.org/)

Also, a [requirements](requirements.txt) file can be used to set up your environment.


### Supported Python Versions

Currently, MCT is being tested on various Python versions:



| Python Version                                                                                                                                                                                                                                                            |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [![Run Tests - Python 3.10](https://github.com/sony/model_optimization/actions/workflows/run_tests_suite_python310.yml/badge.svg)](https://github.com/sony/model_optimization/actions/workflows/run_tests_suite_python310.yml) |
| [![Run Tests - Python 3.9](https://github.com/sony/model_optimization/actions/workflows/run_tests_suite_python39.yml/badge.svg)](https://github.com/sony/model_optimization/actions/workflows/run_tests_suite_python39.yml)   |
| [![Run Tests - Python 3.8](https://github.com/sony/model_optimization/actions/workflows/run_tests_suite_python38.yml/badge.svg)](https://github.com/sony/model_optimization/actions/workflows/run_tests_suite_python38.yml)   |
| [![Run Tests - Python 3.7](https://github.com/sony/model_optimization/actions/workflows/run_tests_suite_python37.yml/badge.svg)](https://github.com/sony/model_optimization/actions/workflows/run_tests_suite_python37.yml)   |


### Supported NN-Frameworks Versions

Currently, MCT supports compressing models of TensorFlow and PyTorch, and
is tested on various versions:

| TensorFlow Version                                                                                   | PyTorch Version                                                                                          |
|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| ![tests](https://github.com/sony/model_optimization/actions/workflows/run_tests_tf211.yml/badge.svg) | ![tests](https://github.com/sony/model_optimization/actions/workflows/run_tests_torch1_13.yml/badge.svg) |
| ![tests](https://github.com/sony/model_optimization/actions/workflows/run_tests_tf210.yml/badge.svg) | ![tests](https://github.com/sony/model_optimization/actions/workflows/run_tests_torch1_12.yml/badge.svg) |
| ![tests](https://github.com/sony/model_optimization/actions/workflows/run_tests_tf29.yml/badge.svg)  | ![tests](https://github.com/sony/model_optimization/actions/workflows/run_tests_torch1_11.yml/badge.svg) |


### Usage Example 
For an example of how to use the post-training quantization, using Keras,
please use this [link](tutorials/example_keras_mobilenet.py).

For an example using PyTorch, please use this [link](tutorials/example_pytorch_mobilenet_v2.py).

For more examples please see the [tutorials' directory](https://github.com/sony/model_optimization/tree/main/tutorials).


## Results
### Keras
Graph of [MobileNetV2](https://keras.io/api/applications/mobilenet/) accuracy on ImageNet vs average bit-width of weights, using 
single-precision quantization, mixed-precision quantization, and mixed-precision quantization with GPTQ.

<img src="docsrc/images/mbv2_accuracy_graph.png">

For more results, please see [1]

### Pytorch
We quantized classification networks from the torchvision library. 
In the following table we present the ImageNet validation results for these models:

| Network Name              | Float Accuracy  | 8Bit Accuracy   | 
| --------------------------| ---------------:| ---------------:| 
| MobileNet V2 [3]          | 71.886          | 71.444           |                                      
| ResNet-18 [3]             | 69.86           | 69.63           |                                      
| SqueezeNet 1.1 [3]        | 58.128          | 57.678          |                                      



## Contributions
MCT aims at keeping a more up-to-date fork and welcomes contributions from anyone.

*You will find more information about contributions in the [Contribution guide](CONTRIBUTING.md).


## License
[Apache License 2.0](LICENSE.md).

## References 

[1] Habi, H.V., Peretz, R., Cohen, E., Dikstein, L., Dror, O., Diamant, I., Jennings, R.H. and Netzer, A., 2021. [HPTQ: Hardware-Friendly Post Training Quantization. arXiv preprint](https://arxiv.org/abs/2109.09113).

[2] [MobilNet](https://keras.io/api/applications/mobilenet/#mobilenet-function) from Keras applications.

[3] [TORCHVISION.MODELS](https://pytorch.org/vision/stable/models.html) 
