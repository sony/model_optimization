# Tutorials

## Table of Contents
- [Introduction](#introduction)
- [Keras Tutorials](#keras-tutorials)
  - [Post-Training Quantization](#keras-ptq)
  - [Gradient-Based Post-Training Quantization](#keras-gptq)
  - [Quantization-Aware Training](#keras-qat)
  - [Structured Pruning](#keras-pruning)
  - [Export Quantized Models](#keras-export)
  - [Debug Tools](#keras-debug)
- [Pytorch Tutorials](#pytorch-tutorials)
  - [Quick-Start with Torchvision](#pytorch-quickstart-torchvision)
  - [Post-Training Quantization](#pytorch-ptq)
  - [Quantization-Aware Training](#pytorch-qat)
  - [Data Generation](#pytorch-data-generation)
  - [Export Quantized Models](#pytorch-export)

## Introduction
Dive into the Model-Compression-Toolkit (MCT) with our collection of tutorials, covering a wide 
range of compression techniques for Keras and Pytorch models. We provide
both Python scripts and interactive Jupyter notebooks for an
engaging and hands-on experience.


## Keras Tutorials

<details id="keras-ptq">
  <summary>Post-Training Quantization (PTQ)</summary>
  
  | Tutorial                     | Included Features                                                                                   |
  |------------------------------|-----------------------------------------------------------------------------------------------------|
  | [MobileNetV2](keras/ptq/example_keras_imagenet.ipynb)  | &#x2705; PTQ                                                                                        |
  | [Mixed-Precision MobileNetV2](tutorials/notebooks/keras/ptq/example_keras_mobilenet_mixed_precision.ipynb) | &#x2705; PTQ <br/> &#x2705; Mixed-Precision                                                         |
  | [Nanodet-Plus](tutorials/notebooks/keras/ptq/example_keras_nanodet_plus.ipynb)             | &#x2705; PTQ                                                                                        |
  | [YoloV8-nano](tutorials/notebooks/keras/ptq/example_keras_yolov8n.ipynb)              | &#x2705; PTQ                                                                                        |
  | [EfficientDetLite0](tutorials/notebooks/keras/ptq/example_keras_effdet_lite0.ipynb) | &#x2705; PTQ <br/> &#x2705; [sony-custom-layers](https://github.com/sony/custom_layers) integration |

</details>

<details id="keras-gptq">
  <summary>Gradient-Based Post-Training Quantization (GPTQ)</summary>

  | Tutorial                     | Included Features       |
  |------------------------------|---------------|
  | [MobileNetV2](tutorials/notebooks/keras/gptq/example_keras_mobilenet_gptq.ipynb)           | &#x2705; GPTQ |

</details>

<details id="keras-qat">
  <summary>Quantization-Aware Training (QAT)</summary>
  
  | Tutorial                                                                          | Included Features      |
  |-----------------------------------------------------------------------------------|--------------|
  | [QAT on MNIST](tutorials/notebooks/keras/gptq/example_keras_mobilenet_gptq.ipynb) | &#x2705; QAT |

</details>


<details id="keras-pruning">
  <summary>Structured Pruning</summary>

  | Tutorial                                                            | Included Features          |
  |---------------------------------------------------------------------|------------------|
  | [Fully-Connected Model Pruning](tutorials/notebooks/keras/pruning/example_keras_pruning_mnist.ipynb) | &#x2705; Pruning |

</details>

<details id="keras-export">
  <summary>Export Quantized Models</summary>

  | Tutorial                                                                              | Included Features         |
  |---------------------------------------------------------------------------------------|-----------------|
  | [Exporter Usage](tutorials/notebooks/keras/export/example_keras_export.ipynb) | &#x2705; Export |
  
</details>

<details id="keras-debug">
  <summary>Debug Tools</summary>

  | Tutorial                                                                            | Included Features       |
  |-------------------------------------------------------------------------------------|-------------------------|
  | [Network Editor Usage](tutorials/notebooks/keras/debug_tools/example_keras_network_editor.ipynb) | &#x2705; Network Editor |
  
</details>

## Pytorch Tutorials


<details id="pytorch-quickstart-torchvision">
  <summary>Quick-Start with Torchvision</summary>
  
  | Tutorial                                                                                                        |
  |-----------------------------------------------------------------------------------------------------------------|
  | [Quick Start - Torchvision Pretrained Model](tutorials/notebooks/pytorch/example_quick_start_torchvision.ipynb) |

</details>


<details id="pytorch-ptq">
  <summary>Post-Training Quantization (PTQ)</summary>
  
  | Tutorial                                                                                                                              | Included Features                                                                                   |
  |---------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
  | [Training & Quantizing Model on MNIST](tutorials/notebooks/pytorch/ptq/example_pytorch_quantization_mnist.ipynb)                      | &#x2705; PTQ                                                                                        |
  | [Mixed-Precision MobileNetV2 on Cifar100](tutorials/notebooks/pytorch/ptq/example_pytorch_mobilenetv2_cifar100_mixed_precision.ipynb) | &#x2705; PTQ <br/> &#x2705; Mixed-Precision                                                         |
  | [SSDLite MobileNetV3 Quantization](tutorials/notebooks/pytorch/ptq/example_pytorch_ssdlite_mobilenetv3.ipynb)                                    | &#x2705; PTQ                                                                                        |

</details>



<details id="pytorch-qat">
  <summary>Quantization-Aware Training (QAT)</summary>
  
  | Tutorial                                                                          | Included Features      |
  |-----------------------------------------------------------------------------------|--------------|
  | [QAT on MNIST](tutorials/notebooks/pytorch/qat/example_pytorch_qat.py) | &#x2705; QAT |

</details>

<details id="pytorch-data-generation">
  <summary>Data Generation</summary>
  
  | Tutorial                                                                                                                          | Included Features                                                                 |
  |-----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
  | [Data-Free Quantization using Data Generation](tutorials/notebooks/pytorch/data_generation/example_pytorch_data_generation.ipynb) | &#x2705; PTQ <br/> &#x2705; Data-Free Quantization <br/> &#x2705; Data Generation |

</details>


<details id="pytorch-export">
  <summary>Export Quantized Models</summary>

  | Tutorial                                                                              | Included Features         |
  |---------------------------------------------------------------------------------------|-----------------|
  | [Exporter Usage](tutorials/notebooks/pytorch/export/example_pytorch_export.ipynb) | &#x2705; Export |
  
</details>

