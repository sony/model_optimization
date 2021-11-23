.. _ug-index:

============================================
Model Compression Toolkit User Guide
============================================


Overview
========

Model Compression Toolkit (MCT) is an open source project for neural networks optimization that enables users to compress and quantize models.
This project enables researchers, developers and engineers an easily way to optimized and quantized state-of-the-art neural network.

Currently, MCT supports hardware-friendly post training quantization (HPTQ) with Tensorflow 2 [1].


MCT project is developed by researchers and engineers working in Sony Semiconductors Israel.

Install
====================================
See the MCT install guide for the pip package, and build from source.


From Source:
::

    git clone https://github.com/sony/model_optimization.git
    python setup.py install


From PyPi - latest stable release:
::

    pip install model-compression-toolkit


A nightly version is also available (unstable):
::

    pip install mct-nightly

Supported Features
====================================

Quantization:

* :ref:`Hardware-friendly Post Training Quantization<ug-keras_post_training_quantization>` [1]
* :ref:`Gradient base post training using knowledge distillation<ug-GradientPTQConfig>` (Experimental)
* :ref:`Mixed-precision post training quantization<ug-keras_post_training_quantization_mixed_precision>` (Experimental)

Visualization:

.. toctree::
    :titlesonly:
    :maxdepth: 1

    Visualize a model and other data within the TensorBoard UI. <../guidelines/visualization>


Quickstart
====================================
Take a look of how you can start using MCT in just a few minutes

.. toctree::
    :titlesonly:
    :maxdepth: 1

    Quick start tutorial for Keras Post Training Quantization<../guidelines/quickstart_keras>





API Documentation
==================
Please visit the MCT API documentation here

.. toctree::
    :titlesonly:
    :maxdepth: 1

    API Documentation<../api/api_docs/index>

References
====================================

[1] Habi, H.V., Peretz, R., Cohen, E., Dikstein, L., Dror, O., Diamant, I., Jennings, R.H. and Netzer, A., 2021. `HPTQ: Hardware-Friendly Post Training Quantization. arXiv preprint. <https://arxiv.org/abs/2109.09113>`_

