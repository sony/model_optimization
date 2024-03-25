# MCT Features
This tutorial set introduces the various quantization tools offered by MCT. 
The notebooks included here illustrate the setup and usage of both basic and advanced post-training quantization methods. 
You'll learn how to refine PTQ (Post-Training Quantization) settings, export models, and explore advanced compression 
techniques such as GPTQ (Gradient-Based Post-Training Quantization), Mixed precision quantization and more.
These techniques are essential for further optimizing models and achieving superior performance in deployment scenarios.

### Keras Tutorials

<details id="keras-ptq">
  <summary>Post-Training Quantization (PTQ)</summary>
  
  | Tutorial                     | Included Features                                                                                   |
  |------------------------------|-----------------------------------------------------------------------------------------------------|
  | [MobileNetV2](keras/ptq/example_keras_imagenet.ipynb)  | &#x2705; PTQ                                                                                        |
  | [Mixed-Precision MobileNetV2](keras/ptq/example_keras_mobilenet_mixed_precision.ipynb) | &#x2705; PTQ <br/> &#x2705; Mixed-Precision                                                         |
  | [Nanodet-Plus](keras/ptq/example_keras_nanodet_plus.ipynb)             | &#x2705; PTQ                                                                                        |
  | [YoloV8-nano](keras/ptq/example_keras_yolov8n.ipynb)              | &#x2705; PTQ                                                                                        |
  | [EfficientDetLite0](keras/ptq/example_keras_effdet_lite0.ipynb) | &#x2705; PTQ <br/> &#x2705; [sony-custom-layers](https://github.com/sony/custom_layers) integration |

</details>

<details id="keras-gptq">
  <summary>Gradient-Based Post-Training Quantization (GPTQ)</summary>

  | Tutorial                     | Included Features       |
  |------------------------------|---------------|
  | [MobileNetV2](keras/gptq/example_keras_mobilenet_gptq.ipynb)           | &#x2705; GPTQ |

</details>

<details id="keras-qat">
  <summary>Quantization-Aware Training (QAT)</summary>
  
  | Tutorial                                          | Included Features      |
  |---------------------------------------------------|--------------|
  | [QAT on MNIST](keras/qat/example_keras_qat.ipynb) | &#x2705; QAT |

</details>


<details id="keras-pruning">
  <summary>Structured Pruning</summary>

  | Tutorial                                                            | Included Features          |
  |---------------------------------------------------------------------|------------------|
  | [Fully-Connected Model Pruning](keras/pruning/example_keras_pruning_mnist.ipynb) | &#x2705; Pruning |

</details>

<details id="keras-export">
  <summary>Export Quantized Models</summary>

  | Tutorial                                                                              | Included Features         |
  |---------------------------------------------------------------------------------------|-----------------|
  | [Exporter Usage](keras/export/example_keras_export.ipynb) | &#x2705; Export |
  
</details>

<details id="keras-debug">
  <summary>Debug Tools</summary>

  | Tutorial                                                                            | Included Features       |
  |-------------------------------------------------------------------------------------|-------------------------|
  | [Network Editor Usage](keras/debug_tools/example_keras_network_editor.ipynb) | &#x2705; Network Editor |
  
</details>

### Pytorch Tutorials


<details id="pytorch-quickstart-torchvision">
  <summary>Quick-Start with Torchvision</summary>
  
  | Tutorial                                                                                                        |
  |-----------------------------------------------------------------------------------------------------------------|
  | [Quick Start - Torchvision Pretrained Model](pytorch/example_quick_start_torchvision.ipynb) |

</details>


<details id="pytorch-ptq">
  <summary>Post-Training Quantization (PTQ)</summary>
  
  | Tutorial                                                                                                                              | Included Features                                                                                   |
  |---------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
  | [Training & Quantizing Model on MNIST](pytorch/ptq/example_pytorch_quantization_mnist.ipynb)                      | &#x2705; PTQ                                                                                        |
  | [Mixed-Precision MobileNetV2 on Cifar100](pytorch/ptq/example_pytorch_mobilenetv2_cifar100_mixed_precision.ipynb) | &#x2705; PTQ <br/> &#x2705; Mixed-Precision                                                         |
  | [SSDLite MobileNetV3 Quantization](pytorch/ptq/example_pytorch_ssdlite_mobilenetv3.ipynb)                                    | &#x2705; PTQ                                                                                        |

</details>



</details>

<details id="pytorch-pruning">
  <summary>Structured Pruning</summary>

  | Tutorial                                                                             | Included Features          |
  |--------------------------------------------------------------------------------------|------------------|
  | [Fully-Connected Model Pruning](pytorch/pruning/example_pytorch_pruning_mnist.ipynb) | &#x2705; Pruning |


</details>

<details id="pytorch-data-generation">
  <summary>Data Generation</summary>
  
  | Tutorial                                                                                                                          | Included Features                                                                 |
  |-----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
  | [Data-Free Quantization using Data Generation](pytorch/data_generation/example_pytorch_data_generation.ipynb) | &#x2705; PTQ <br/> &#x2705; Data-Free Quantization <br/> &#x2705; Data Generation |

</details>


<details id="pytorch-export">
  <summary>Export Quantized Models</summary>

  | Tutorial                                                                              | Included Features         |
  |---------------------------------------------------------------------------------------|-----------------|
  | [Exporter Usage](pytorch/export/example_pytorch_export.ipynb) | &#x2705; Export |
  
</details>
