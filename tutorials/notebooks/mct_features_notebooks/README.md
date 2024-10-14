# MCT Features
This tutorial set introduces the various quantization tools offered by MCT. 
The notebooks included here illustrate the setup and usage of both basic and advanced post-training quantization methods. 
You'll learn how to refine PTQ (Post-Training Quantization) settings, export models, and explore advanced compression 
techniques such as GPTQ (Gradient-Based Post-Training Quantization), Mixed precision quantization and more.
These techniques are essential for further optimizing models and achieving superior performance in deployment scenarios.

### Keras Tutorials

<details id="keras-ptq">
  <summary>Post-Training Quantization (PTQ)</summary>
  
  | Tutorial                                                                                   | Included Features                                                                                   |
  |--------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
  | [Basic Post-Training Quantization (PTQ)](keras/example_keras_post-training_quantization.ipynb)          | &#x2705; PTQ                                                                                        |
  | [MobileNetV2](../imx500_notebooks/keras/example_keras_mobilenetv2_for_imx500.ipynb)        | &#x2705; PTQ                                                                                        |
  | [Mixed-Precision MobileNetV2](keras/example_keras_mobilenet_mixed_precision.ipynb)         | &#x2705; PTQ <br/> &#x2705; Mixed-Precision                                                         |
  | [Nanodet-Plus](../imx500_notebooks/keras/example_keras_nanodet_plus_for_imx500.ipynb)      | &#x2705; PTQ                                                                                        |
  | [EfficientDetLite0](../imx500_notebooks/keras/example_keras_effdet_lite0_for_imx500.ipynb) | &#x2705; PTQ <br/> &#x2705; [sony-custom-layers](https://github.com/sony/custom_layers) integration |

</details>

<details id="keras-gptq">
  <summary>Gradient-Based Post-Training Quantization (GPTQ)</summary>

  | Tutorial                     | Included Features       |
  |------------------------------|---------------|
  | [MobileNetV2](keras/example_keras_mobilenet_gptq.ipynb)           | &#x2705; GPTQ |

</details>

<details id="keras-qat">
  <summary>Quantization-Aware Training (QAT)</summary>
  
  | Tutorial                                          | Included Features      |
  |---------------------------------------------------|--------------|
  | [QAT on MNIST](keras/example_keras_qat.ipynb) | &#x2705; QAT |

</details>


<details id="keras-pruning">
  <summary>Structured Pruning</summary>

  | Tutorial                                                            | Included Features          |
  |---------------------------------------------------------------------|------------------|
  | [Fully-Connected Model Pruning](keras/example_keras_pruning_mnist.ipynb) | &#x2705; Pruning |

</details>

<details id="keras-export">
  <summary>Export Quantized Models</summary>

  | Tutorial                                                                              | Included Features         |
  |---------------------------------------------------------------------------------------|-----------------|
  | [Exporter Usage](keras/example_keras_export.ipynb) | &#x2705; Export |
  
</details>

<details id="keras-debug">
  <summary>Debug Tools</summary>

  | Tutorial                                                                            | Included Features       |
  |-------------------------------------------------------------------------------------|-------------------------|
  | [Network Editor Usage](keras/example_keras_network_editor.ipynb) | &#x2705; Network Editor |
  
</details>

### Pytorch Tutorials


<details id="pytorch-ptq">
  <summary>Post-Training Quantization (PTQ)</summary>
  
  | Tutorial                                                                                                  | Included Features                           |
  |-----------------------------------------------------------------------------------------------------------|---------------------------------------------|
  | [Basic Post-Training Quantization (PTQ)](pytorch/example_pytorch_post_training_quantization.ipynb)        | &#x2705; PTQ                                |
  | [Mixed-Precision Post-Training Quantization](pytorch/example_pytorch_mixed_precision_ptq.ipynb)           | &#x2705; PTQ <br/> &#x2705; Mixed-Precision |
  | [Advanced Gradient-Based Post-Training Quantization (GPTQ)](pytorch/example_pytorch_mobilenet_gptq.ipynb) | &#x2705; GPTQ                               |

</details>

<details id="pytorch-pruning">
  <summary>Structured Pruning</summary>

  | Tutorial                                                                             | Included Features          |
  |--------------------------------------------------------------------------------------|------------------|
  | [Fully-Connected Model Pruning](pytorch/example_pytorch_pruning_mnist.ipynb) | &#x2705; Pruning |


</details>

<details id="pytorch-data-generation">
  <summary>Data Generation</summary>
  
  | Tutorial                                                                                            | Included Features                                                                                    |
  |-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
  | [Zero-Shot Quantization (ZSQ) using Data Generation](pytorch/example_pytorch_data_generation.ipynb) | &#x2705; PTQ <br/> &#x2705; ZSQ <br/> &#x2705; Data-Free Quantization <br/> &#x2705; Data Generation |

</details>


<details id="pytorch-export">
  <summary>Export Quantized Models</summary>

  | Tutorial                                                                              | Included Features         |
  |---------------------------------------------------------------------------------------|-----------------|
  | [Exporter Usage](pytorch/example_pytorch_export.ipynb) | &#x2705; Export |
  
</details>
<details id="pytorch-xquant">
  <summary>Quantization Troubleshooting</summary>

  | Tutorial                                                                                       | Included Features |
  |------------------------------------------------------------------------------------------------|-------------------|
  | [Quantization Troubleshooting using the Xquant Feature](pytorch/example_pytorch_xquant.ipynb) | &#x2705; Debug    |
  
</details>
