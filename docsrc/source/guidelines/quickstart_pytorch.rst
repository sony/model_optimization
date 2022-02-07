:orphan:

.. _ug-quickstart-pytorch:

=================================================
MCT Quickstart Guideline for Pytorch models
=================================================

Here is an example of a code that shows how to use MCT with Pytorch models.


Import MCT and mobilenet_v2 from torchvision.models:

.. literalinclude:: ../../../tutorials/example_pytorch_mobilenet_v2.py
    :language: python
    :lines: 15-16

|

Data preprocessing functions:

.. literalinclude:: ../../../tutorials/example_pytorch_mobilenet_v2.py
    :language: python
    :lines: 17-27

|

Initialize data loader:

.. literalinclude:: ../../../tutorials/example_pytorch_mobilenet_v2.py
    :language: python
    :lines: 32-63

|

Run Post Training Quantization:

.. literalinclude:: ../../../tutorials/example_pytorch_mobilenet_v2.py
    :language: python
    :lines: 65-70

