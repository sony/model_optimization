# MCT Installation

## Setting up work environment

### From PyPi - latest stable release
See the MCT install guide for the pip package.
```
pip install model-compression-toolkit
```

A nightly package is also available (unstable):
```
pip install mct-nightly
```

### Working from source code

Clone the repository and install the required packages (via [requirements](requirements.txt)).
```
git clone https://github.com/sony/model_optimization.git
cd model_optimization
pip install -r requirements.txt
```

## Requirements

To run MCT, one of the supported frameworks, either Tensorflow or Pytorch, needs to be installed.

For use with Tensorflow please install the packages: 
[tensorflow](https://www.tensorflow.org/install), 
[tensorflow-model-optimization](https://www.tensorflow.org/model_optimization/guide/install)

For use with PyTorch please install the package: 
[torch](https://pytorch.org/)
