# Model Compression Toolkit (MCT)
![tests](https://github.com/sony/model_optimization/actions/workflows/run_tests_suite.yml/badge.svg)


Model Compression Toolkit (MCT) is an open source project for neural network model optimization under efficient hardware constrained. <br />
This project enables researchers, developers and engineers an easy way to optimized and deploy state-of-the-art neural network on efficient hardware. <br />
Specifically this project apply constrained quantization and pruning scheme on a neural network. 

Currently, this project only support hardware friendly post training quantization (HPTQ) with Tensorflow 2 [1]. 

MCT project is developed by researchers and engineers working on the Sony Semiconductors Israel.

For more information, please visit our [project website](https://sony.github.io/model_optimization/).

## Table of Contents

- [Getting Started](#getting-started)
- [Supported features](#supported-features)
- [Results](#results)
- [Contributions](#contributions)
- [License](#license)

## Getting Started

This section provides a quick starting guide start with installtion via source code or pip server then a short example usage.

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


### Example Usage
Here is a snapshot of a code that shown an example of how to use the post training quantization using keras.

```python
import model_compression_toolkit as mct

# Set the batch size of the images at each calibration iteration.
batch_size = 50

# Create a representative data generator, which returns a list of images.
# Load a folder of images. 
folder = '/path/to/images/folder'

# The images can be preprocessed using a list of preprocessing functions.
def normalization(x):
    return (x - 127.5) / 127.5

# Create a FolderImageLoader instance which loads the images, preprocess them and enables you to sample batches of them.
image_data_loader = mct.FolderImageLoader(folder,
                                          preprocessing=[normalization],
                                          batch_size=batch_size)

# Create a Callable representative dataset for calibration purposes.
# The function should be called without any arguments, and should return a list numpy arrays (array for each model's input).
# For example: if the model has two input tensors - one with input shape of 32X32X3 and the second with input 
# shape of 224X224X3, and we calibrate the model using batches of 20 images,
# calling representative_data_gen() should return a list 
# of two numpy.ndarray objects where the arrays' shapes are [(20, 32, 32, 3), (20, 224, 224, 3)].
def representative_data_gen() -> list:
        return [image_data_loader.sample()]


# Create a model and quantize it using the representative_data_gen as the calibration images.
# Set the number of calibration iterations to 10.
quantized_model, quantization_info = mct.keras_post_training_quantization(model,
                                                                          representative_data_gen,
                                                                          n_iter=10)
```
For more example please see the tutorials' directory.


## Supported Features

Quantization:

	* Post Training Quantization 
    * Gradient base post training (Experimental) 
    
Tensorboard Visualization (Experimental):

    * CS Analyizer: compare comprased model with orignal model to analysis large accuracy drop.
    * Activation statisicis and errors
     

Note that currently we only have full support for Keras layers, using the TensorFlow native layer may lead to unexpected behavior. This limitation will be removed in future releases. 

MCT is test with Tensorflow Version 2.5. 

## Tutorials and Results
As part of the MCT library, we have a set of example network on image classification which can be used as an example while using the package.

* Image Classification Example with MobileNet V1 on ImageNet dataset

| Network Name             | Float Accuracy  | 8Bit Accuracy   | Comments                             |
| -------------------------| ---------------:| ---------------:| ------------------------------------:|
| MobileNetV1 [2]          | 70.558          | 70.418          |                                      |


For more results please see [1]

## Contributions
MCT aims at keeping a more up-to-date fork and welcomes contributions from anyone.

*You will find more information about contributions in the [Contribution guide](CONTRIBUTING.md).


## License
[Apache License 2.0](LICENSE).

## Refernce 

[1] Habi, H.V., Peretz, R., Cohen, E., Dikstein, L., Dror, O., Diamant, I., Jennings, R.H. and Netzer, A., 2021. [HPTQ: Hardware-Friendly Post Training Quantization. arXiv preprint](https://arxiv.org/abs/2109.09113).

[2] [MobilNet](https://keras.io/api/applications/mobilenet/#mobilenet-function) from Keras applications.
