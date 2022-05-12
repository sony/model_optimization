:orphan:

.. _ug-quickstart-pytorch:

=================================================
MCT Quickstart Guideline for Pytorch models
=================================================

Here is an example of a code that shows how to use MCT with Pytorch models.


Import MCT and mobilenet_v2 from torchvision.models:

.. literalinclude:: ../../../tutorials/example_pytorch_mobilenet_v2.py
    :language: python
    :lines: 20-23

|

Data preprocessing functions:

.. literalinclude:: ../../../tutorials/example_pytorch_mobilenet_v2.py
    :language: python
    :lines: 23-30

|

Initialize data loader:

.. literalinclude:: ../../../tutorials/example_pytorch_mobilenet_v2.py
    :language: python
    :lines: 33-66

|

Get a TargetPlatformCapabilities:

.. literalinclude:: ../../../tutorials/example_pytorch_mobilenet_v2.py
    :language: python
    :lines: 66-72

|


Run Post Training Quantization:

.. literalinclude:: ../../../tutorials/example_pytorch_mobilenet_v2.py
    :language: python
    :lines: 73-87

