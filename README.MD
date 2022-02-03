# Model Compression Toolkit (MCT)
![tests](https://github.com/sony/model_optimization/actions/workflows/run_tests_suite_tf27.yml/badge.svg)


Model Compression Toolkit (MCT) is an open source project for neural network model optimization under efficient, constrained hardware. <br />
This project provides researchers, developers and engineers tools for optimizing and deploying state-of-the-art neural network on efficient hardware. <br />
Specifically, this project applies constrained quantization and pruning schemes on a neural network. 

Currently, this project only supports hardware friendly post training quantization (HPTQ) with Tensorflow 2 [1]. 

The MCT project is developed by researchers and engineers working at Sony Semiconductors Israel.

For more information, please visit our [project website](https://sony.github.io/model_optimization/).

## Table of Contents

- [Getting Started](#getting-started)
- [Supported features](#supported-features)
- [Results](#results)
- [Contributions](#contributions)
- [License](#license)

## Getting Started

This section provides a quick starting guide. We begin with installtion via source code or pip server. Then, we provide a short usage example.

### Installation
See the MCT install guide for the pip package, and build from source.


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

To run MCT, one of the supported frameworks, Tenosflow/Pytorch, needs to be installed.

For using with Tensorflow please install the packages: 
[tensorflow](https://www.tensorflow.org/install), 
[tensorflow-model-optimization](https://www.tensorflow.org/model_optimization/guide/install)

For using with Pytorch please install the packages: 
[torch](https://pytorch.org/)
### Usage Example 
For an example of how to use the post training quantization, using Keras,
please use this [link](tutorials/example_keras_mobilenet.py).

For an example using Pytorch, please use this [link](tutorials/example_pytorch_mobilenet_v2.py).

For more examples please see the [tutorials' directory](tutorials).


## Supported Features

Quantization:

	* Post Training Quantization 
    * Gradient based post training (Experimental, Keras only) 
    * Mixed-precision post training quantization (Experimental, Keras only)
    
Tensorboard Visualization (Experimental):

    * CS Analyizer: compare a model compressed with the orignal model to analyze large accuracy drops.
    * Activation statisicis and errors

MCT is tested with Tensorflow Version 2.7. 

## Results
### Keras
As part of the MCT library, we have a set of example networks on image classification. These networks can be used as examples when using the package.

* Image Classification Example with MobileNet V1 on ImageNet dataset

| Network Name             | Float Accuracy  | 8Bit Accuracy   | Comments                             |
| -------------------------| ---------------:| ---------------:| ------------------------------------:|
| MobileNetV1 [2]          | 70.558          | 70.418          |                                      |


For more results please see [1]

### Pytorch
We quantized classification networks from the torchvision library. 
In the following table we present the ImageNet validation results for these models:

| Network Name              | Float Accuracy  | 8Bit Accuracy   | 
| --------------------------| ---------------:| ---------------:| 
| MobileNet V2 [3]          | 71.886          | 71.804          |                                      
| ResNet-18 [3]             | 69.86           | 69.74           |                                      
| SqueezeNet 1.1 [3]        | 58.128          | 57.848          |                                      



## Contributions
MCT aims at keeping a more up-to-date fork and welcomes contributions from anyone.

*You will find more information about contributions in the [Contribution guide](CONTRIBUTING.md).


## License
[Apache License 2.0](LICENSE).

## References 

[1] Habi, H.V., Peretz, R., Cohen, E., Dikstein, L., Dror, O., Diamant, I., Jennings, R.H. and Netzer, A., 2021. [HPTQ: Hardware-Friendly Post Training Quantization. arXiv preprint](https://arxiv.org/abs/2109.09113).

[2] [MobilNet](https://keras.io/api/applications/mobilenet/#mobilenet-function) from Keras applications.

[3] [TORCHVISION.MODELS](https://pytorch.org/vision/stable/models.html) 
