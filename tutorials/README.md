# MCT Tutorials
Explore the Model Compression Toolkit (MCT) through our tutorials, 
covering compression techniques for Keras and PyTorch models. 
Access interactive Jupyter notebooks for hands-on learning.


## Getting started
Learn how to quickly quantize pre-trained models using MCT's post-training quantization technique for both Keras and PyTorch models.
- [Keras MobileNetV2 post training quantization](notebooks/keras/ptq/example_keras_imagenet.ipynb)
- [Pytorch MobileNetV2 post training quantization](notebooks/pytorch/ptq/example_pytorch_quantization_mnist.ipynb)

## MCT Features
This set of tutorials covers all the quantization tools provided by MCT. 
The notebooks in this section demonstrate how to configure and run simple and advanced post-training quantization methods.
This includes fine-tuning PTQ (Post-Training Quantization) configurations, exporting models,
and exploring advanced compression techniques. 
These techniques are essential for further optimizing models and achieving superior performance in deployment scenarios.
- [MCT notebooks](notebooks/MCT_notebooks.md)

## Quantization for Sony-IMX500 deployment

This section provides several guides on quantizing pre-trained models to meet specific constraints for deployment on the
[Sony-IMX500](https://developer.sony.com/imx500/) processing platform. 
We will cover various tasks and demonstrate the necessary steps to achieve efficient quantization for optimal
deployment performance.
- [IMX500 notebooks](notebooks/IMX500_notebooks.md)
