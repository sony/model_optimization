![Sony Semiconductor Israel, Inc.](docs/images/ssi_logo.png)


# Sony Networks Optimization Package (SMOP)
Sony Networks Optimization Package (SMOP) is an open source project for neural network model optimization under efficient hardware constrained. <br />
This project enables researchers, developers and engineers an easy way to optimized and deploy state-of-the-art neural network on efficient hardware. <br />
Specifically this project apply constrained quantization and pruning scheme on a neural network. 

Currently, this project only support hardware friendly post training quantization (HPTQ) with Tensorflow 2 [1]. 

SMOP project is developed by researchers and engineers working on the Sony Semiconductors Israel.

## Table of Contents

- [Getting Started](#getting-started)
- [Supported features](#supported-features)
- [Results](#results)
- [Contributions](#contributions)
- [License](#license)

## Getting Started

This section provides a quick starting guide start with installation via source code and a short example usage.

### Installation
See the SMOP install guide for build from source.


#### From Source
```
git clone https://github.com/sony/model_optimization.git
python setup.py install
```

### Example Usage
Here is a snapshot of a code that shown an example of how to use the post training quantization using keras.

```python
import sony_model_optimization_package as smop

def representative_data_gen() -> list:
        return [image_data_loader.sample()]


quantized_model, quantization_info = smop.keras_post_training_quantization(model,
                                                                           representative_data_gen,
                                                                           n_iter=200)
```
For more example please see the tutorials' directory.

## Supported Features

Quantization:

	* Post Training Quantization 
    * Gradient base post training using knowledge distillation (Experimental) 
    
Tensorboard Visualization (Experimental):

    * CS Analyizer: compare comprased model with orignal model to analysis large accuracy drop.
    * Activation statisicis and errors
     

Note that currently we only have full support for Keras layers, using the TensorFlow native layer may lead to unexpected behavior. This limitation will be removed in future releases. 

SMOP was tested with Tensorflow Version 2.5. 

## Tutorials and Results
As part of the SMOP library, we have a set of example network on image classification which can be used as an example while using the package.

* Image Classification Example with MobileNet V1 on ImageNet dataset

| Network Name             | Float Accuracy  | 8Bit Accuracy   | Comments                             |
| -------------------------| ---------------:| ---------------:| ------------------------------------:|
| MobileNetV1 [2]          | 70.558          | 70.418          |                                      |


For more results please see [1]

## Contributions
SMOP aims at keeping a more up-to-date fork and welcomes contributions from anyone.

*You will find more information about contributions in the [Contribution guide](CONTRIBUTING.md).


## License
Sony Networks Optimization Package (SMOP) is provided under the [BSD 3.0](LICENSE) license.

## Refernce 

[1] Habi, H.V., Peretz, R., Cohen, E., Dikstein, L., Dror, O., Diamant, I., Jennings, R.H. and Netzer, A., 2021. HPTQ: Hardware-Friendly Post Training Quantization. arXiv preprint.

[2] [MobilNet](https://keras.io/api/applications/mobilenet/#mobilenet-function) from Keras applications.
