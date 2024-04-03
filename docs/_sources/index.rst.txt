.. _ug-index:

============================================
Model Compression Toolkit User Guide
============================================


Overview
========

Model Compression Toolkit (MCT) is an open source project for neural networks optimization that enables users to compress and quantize models.
This project enables researchers, developers and engineers an easily way to optimized and quantized state-of-the-art neural network.

MCT project is developed by researchers and engineers working in Sony Semiconductor Israel.

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

For using with Tensorflow please install the packages:
`tensorflow <https://www.tensorflow.org/install>`_


For using with Pytorch please install the package:
`torch <https://pytorch.org/>`_


Supported Features
====================================

Keras:

* :ref:`Post Training Quantization<ug-keras_post_training_quantization>` [1]
* :ref:`Gradient based post training using knowledge distillation<ug-keras_gradient_post_training_quantization>` [2]
* :ref:`Init model for Quantization Aware Training<ug-keras_quantization_aware_training_init_experimental>` (Experimental)
* :ref:`Finalize model after Quantization Aware Training<ug-keras_quantization_aware_training_finalize_experimental>` (Experimental)
* :ref:`Structured pruning<ug-keras_pruning_experimental>` (Experimental)
* :ref:`Data generation<ug-keras_data_generation_experimental>` (Experimental)

Pytorch:

* :ref:`Post Training Quantization<ug-pytorch_post_training_quantization>` [1]
* :ref:`Gradient based post training using knowledge distillation<ug-pytorch_gradient_post_training_quantization>` [2]
* :ref:`Init model for Quantization Aware Training<ug-pytorch_quantization_aware_training_init_experimental>` (Experimental)
* :ref:`Finalize model after Quantization Aware Training<ug-pytorch_quantization_aware_training_finalize_experimental>` (Experimental)
* :ref:`Structured pruning<ug-pytorch_pruning_experimental>` (Experimental)
* :ref:`Data generation<ug-pytorch_data_generation_experimental>` (Experimental)


Visualization:

.. toctree::
    :titlesonly:
    :maxdepth: 1

    Visualize a model and other data within the TensorBoard UI. <../guidelines/visualization>


Quickstart
====================================
Take a look of how you can start using MCT in just a few minutes!

Visit our `notebooks <https://github.com/sony/model_optimization/tree/main/tutorials/notebooks>`_ and `MCT quick start. <https://github.com/sony/model_optimization/tree/main/tutorials/quick_start>`_


API Documentation
==================
Please visit the MCT API documentation here

.. toctree::
    :titlesonly:
    :maxdepth: 1

    API Documentation<../api/api_docs/index>

Technical Constraints
=========================

* MCT doesn't keep the structure of the model's output. For example, if the output of a model is a list of lists of Tensors [[out1, out2], out3], the optimized model output will be [out1, out2, out3]

References
====================================

[1] Habi, H.V., Peretz, R., Cohen, E., Dikstein, L., Dror, O., Diamant, I., Jennings, R.H. and Netzer, A., 2021. `HPTQ: Hardware-Friendly Post Training Quantization. arXiv preprint. <https://arxiv.org/abs/2109.09113>`_

[2] Gordon, O., Habi, H.V., and Netzer, A., 2023. `EPTQ: Enhanced Post-Training Quantization via Label-Free Hessian. arXiv preprint. <https://arxiv.org/abs/2309.11531>`_
