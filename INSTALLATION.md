# MCT Installation

## Setting up work environment

### From PyPi - latest stable release
To install the latest stable release of MCT from PyPi,
follow the MCT install guide and run the following command:
```
pip install model-compression-toolkit
```

If you prefer to use the nightly package (unstable version),
you can install it with the following command:
```
pip install mct-nightly
```

### Working from Source Code
To work with the MCT source code, follow these steps:

1. Clone the repository:
```
git clone https://github.com/sony/model_optimization.git
cd model_optimization

```
2. Install the required packages listed in the requirements file:
```
pip install -r requirements.txt
```


## Requirements

Before running MCT, make sure to install one of the supported frameworks: TensorFlow or PyTorch.

If you intend to use MCT with TensorFlow, install the following packages: 
[tensorflow](https://www.tensorflow.org/install), 
[tensorflow-model-optimization](https://www.tensorflow.org/model_optimization/guide/install)

If you plan to use MCT with PyTorch, install the following package: 
[torch](https://pytorch.org/)


## Troubleshooting
If you encounter any issues during installation, please open an issue.