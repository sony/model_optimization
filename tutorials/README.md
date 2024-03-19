# MCT Tutorials

Dive into the Model-Compression-Toolkit (MCT) with our collection of tutorials, covering a wide 
range of compression techniques for Keras and Pytorch models. We provide interactive Jupyter notebooks for an
engaging and hands-on experience.

## Getting started
This "hello world" notebook shows how to quickly quantize a pre-trained model using MCT post training quantization technique both for Keras models and Pytorch models.
- [Keras MobileNetV2 post training quantization](keras/ptq/example_keras_imagenet.ipynb)
- [Pytorch MobileNetV2 post training quantization](pytorch/ptq/example_pytorch_mobilenet_v2.py)

## MCT Features
In these examples, we will cover more advanced topics related to quantization. 
This includes fine-tuning PTQ (Post-Training Quantization) configurations, exporting models,
and exploring advanced compression techniques. 
These techniques are crucial for optimizing models further and achieving better performance in deployment scenarios.
- [MCT notebooks](notebooks/MCT_notebooks.md)

## Quantization for Sony-IMX500 deployment
This section provides a guide on quantizing pre-trained models to meet specific constraints for deployment on the
processing platform. Our focus will be on quantizing models for deployment on [Sony-IMX500](https://developer.sony.com/imx500/) processing platform. 
We will cover various tasks and demonstrate the necessary steps to achieve efficient quantization for optimal
deployment performance.
- [IMX500 notebooks](notebooks/IMX500_notebooks.md)


