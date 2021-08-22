.. _ug-index:

============================================
Sony Network Optimization Package User Guide
============================================


Overview
========

Sony Network Optimization Package (SNOP) is an open source project for neural network network optimization that enables users to compress and quantize models.
This project enables researchers, developers and engeniners an easily way to optimized and quantized state-of-the-art neural network.

Currently, SNOP support support constrained post training quantization (CPTQ) with Tensorflow 2 [1].

SNOP project is developed by researchers and engineers working in Sony Semiconductor's Israel.

Install
====================================
See the SNOP install guide for the pip package, and build from source.


From PIP:
::

    python setup.py install

From Source:
::

    git clone https://github.com/sony-si/quantization-library.git
    python setup.py install


Supported Features
====================================

Quantization:

* Constraint Post Training Quantization [1]
* Gradient base post training using kowlaged distillation (Experimental)

Visualization:

.. toctree::
    :titlesonly:
    :maxdepth: 1

    Visualize a model and other data within the TensorBoard UI. <../guidelines/visualization>


Quickstart
====================================
Take a look of how you can start using SNOP in just a few minutes

.. toctree::
    :titlesonly:
    :maxdepth: 1

    Quick start tutorial for Keras Post Training Quantization<../guidelines/quickstart_keras>





API Documentation
==================
Please visit the SNOP API documentation here

.. toctree::
    :titlesonly:
    :maxdepth: 1

    API Documentation<../api/api_docs/index>

References
====================================
