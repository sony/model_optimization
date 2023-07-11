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
#### Single model quantization example
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

#### Set of models quantization example
In the following example, we evaluate a set of pre-trained models. Assuming that the necessary packages have been installed (as mentioned earlier), we use the following command:
```python
python main.py --models_list_csv <myfile.csv>
```
The content of the CSV file consists of a tabular structure where each line represents a model along with its corresponding parameters to be used. For instance:

| model_name    | model_library | dataset_name  | validation_dataset_folder     | representative_dataset_folder     |
|---------------|---------------|---------------|-------------------------------|-----------------------------------|
| mobilenet_v2  | torchvision   | ImageNet      | /path/to/validation/dataset   | /path/to/representative/dataset   |
| regnetx_002   | timm          | ImageNet      | /path/to/validation/dataset   | /path/to/representative/dataset   |



Please note that the above example assumes the command is run from the command line and that the [main.py](./main.py) script is in the current directory.

Make sure to refer to the script itself to understand all available parameters and their usage.

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