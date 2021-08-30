.. _ug-index:

============================================
Sony Model Optimization Package User Guide
============================================


Overview
========

Sony Model Optimization Package (SMOP) is an open source project for neural network network optimization that enables users to compress and quantize models.
This project enables researchers, developers and engeniners an easily way to optimized and quantized state-of-the-art neural network.

Currently, SMOP support support hardware-friendly post training quantization (HPTQ) with Tensorflow 2 [1].

SMOP project is developed by researchers and engineers working in Sony Semiconductor's Israel.

Install
====================================
See the SMOP install guide for build from source.



From Source:
::

    git clone https://github.com/sony/model_optimization.git
    python setup.py install


Supported Features
====================================

Quantization:

* Hardware-friendly Post Training Quantization [1]
* Gradient base post training using kowlaged distillation (Experimental)

Visualization:

.. toctree::
    :titlesonly:
    :maxdepth: 1

    Visualize a model and other data within the TensorBoard UI. <../guidelines/visualization>


Quickstart
====================================
Take a look of how you can start using SMOP in just a few minutes

.. toctree::
    :titlesonly:
    :maxdepth: 1

    Quick start tutorial for Keras Post Training Quantization<../guidelines/quickstart_keras>





API Documentation
==================
Please visit the SMOP API documentation here

.. toctree::
    :titlesonly:
    :maxdepth: 1

    API Documentation<../api/api_docs/index>

References
====================================

[1] Habi, H.V., Peretz, R., Cohen, E., Dikstein, L., Dror, O., Diamant, I., Jennings, R.H. and Netzer, A., 2021. HPTQ: Hardware-Friendly Post Training Quantization. arXiv preprint.
