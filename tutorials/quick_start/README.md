# MCT Quick Start 

This example project demonstrates the capabilities of MCT (Model Compression Toolkit) and illustrates its interface
with various model collections libraries. By accessing a wide range of pre-trained models, this project allows users to
generate a quantized version of their chosen model with a single click. 

Currently, the project supports a selection of models from each library. However, our ongoing goal is to continually
expand the support, aiming to include more models
in the future.   


## Supported libraries
- torchvision: [https://pytorch.org/vision/stable/index.html](https://pytorch.org/vision/stable/index.html)
- timm: [https://github.com/huggingface/pytorch-image-models/tree/main/timm](https://github.com/huggingface/pytorch-image-models/tree/main/timm)
- ultralytics: [https://ultralytics.com](https://ultralytics.com)
- keras-applications: [https://keras.io/api/applications](https://keras.io/api/applications/)

## Getting Started
### Installation 
Install one of the supported libraries above.


### Usage Examples
#### Basic model quantization example - Post training quantization (PTQ)
In the following example, we are evaluating the MCT on pre-trained mobilenet_v2, taken from torchvision models library
1. Install the required library using `pip`:
```bash
pip install torchvision
 ```
2. Run main.py script:
```python
python main.py --model_name mobilenet_v2 --model_library torchvision --validation_dataset_folder <my path> --representative_dataset_folder <my path> 
```
In this example, we are running the main.py script with the following parameters:
- `--model_name`: Specifies the name of the model to be used (mobilenet_v2 in this case).
- `--model_library`: Specifies the package of the pre-trained models from which the model name is taken (in this case, torchvision).
- `--validation_dataset_folder`: Specifies the path to the local copy of the dataset to be used for evaluation.
- `--representative_dataset_folder`: Specifies the path to the local copy of the dataset to be used for quantization.

Please note that the above example assumes the command is run from the command line and that the [main.py](./main.py) script is in the current directory.

Make sure to refer to the script itself to understand all available parameters and their usage.

#### Advanced model quantization example
##### Mixed-precision 
In this example, we use the MCT Mixed-Precision quantization workflow to further reduce the model's size, with minimal reduction in the quantized model accuracy. 
We use the same pre-trained mobilenet_v2 model as our baseline, with the goal of attaining a model size
that is 1/5 of the original floating-point model weights size. This is equivalent to a size reduction factor of 5. 
In contrast, the basic PTQ example illustrated earlier only manages to decrease the model size by a factor of 4.

You can execute the following Python script to compress the model:
```python
python main.py --model_name mobilenet_v2 --model_library torchvision --mp_weights_compression 5 --validation_dataset_folder <my path> --representative_dataset_folder <my path> 
```

##### Gradient-based post training quantization 
The following example demontrates the use of MCT's Gradient-based Post-Training Quantization (GPTQ) workflow. 
This approach is superior to the basic PTQ method as it refines the quantized model weights in order to regain performance.
The weights modification is done through a knowledge distillation technique sourced from the original floating-point model.

To execute the model compression with this approach, run the following Python script:
```python
python main.py --model_name mobilenet_v2 --model_library torchvision --gptq --validation_dataset_folder <my path> --representative_dataset_folder <my path> 
```

Please note that the Mixed-Precision and Gradient-based Post Training Quantization (GPTQ) strategies can be combined to achieve a more significant model compression while mitigating the impact on model performance.
#### More examples
More details and examples for using Ultrlytics models can be found in this [readme](./pytorch_fw/ultralytics/README.md)   

## Results
The latest performance results of MCT on various of models can be found in the [model_quantization_results.csv](./results/model_quantization_results.csv) table. 

## External Package Versions

The following external packages were tested with this project:

- torchvision: Version 0.14.0
- timm: Version 0.6.13
- ultralytics: Version 8.0.47
- keras-applications: Version 2.9.0

## License
This project is licensed under [Apache License 2.0](../../LICENSE.md).
However, please note that different external packages have their own licenses. When using this project, you have the option to choose one of the following external packages:

- torchvision: [license link](https://github.com/UiPath/torchvision/blob/master/LICENSE), [license copy](./pytorch_fw/torchvision/LICENSE)
- timm: [license link](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE), [license copy](./pytorch_fw/timm/LICENSE)
- ultralytics: [license link](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), [license copy](./pytorch_fw/ultralytics/LICENSE)
- keras-applications: [license link](https://github.com/keras-team/keras-applications/blob/master/LICENSE), [license copy](./keras_fw/keras_applications/LICENSE)