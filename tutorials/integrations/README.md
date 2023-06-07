# MCT integration example 

This example project demonstrates the capabilities of MCT and illustrates its interface with various model collections
packages. 


## Supported model collections libraries
- torchvision: [https://pytorch.org/vision/stable/index.html](https://pytorch.org/vision/stable/index.html)
- timm: [https://timm.fast.ai](https://timm.fast.ai)
- ultralytics: [https://ultralytics.com](https://ultralytics.com)


## Getting Started
### Installation 
- Install MCT - Please refer to [MCT documentation](https://github.com/sony/model_optimization/blob/main/README.md)
- Install one of the supported packages above.


### Usage Example
In the following example, we are evaluating the MCT on pre-trained mobilenet_v2, taken from torchvision models library
1. Install the required library using `pip`:
```bash
pip install model-compression-toolkit
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

In the next example we are evaluating list of pre-trained models. Here we assume the relevant packages were installed (see above), and run the following command:
```python
python main.py --benchmark_csv_list <myfile.csv>
```
The csv file content is a table where each line represent a model and the parameters to be used, for example:

| model_name    | model_library | dataset_name  | validation_dataset_folder     | representative_dataset_folder     |
|---------------|---------------|---------------|-------------------------------|-----------------------------------|
| mobilenet_v2  | torchvision   | ImageNet      | /path/to/validation/dataset   | /path/to/representative/dataset   |
| regnetx_002   | timm          | ImageNet      | /path/to/validation/dataset   | /path/to/representative/dataset   |



Please note that the above example assumes the command is run from the command line and that the [main.py](https://sony.github.io/model_optimization/benchmark/main.py) script is in the current directory

Make sure to refer to the script itself to understand all available parameters and their usage.
## Results
The latest MCT benchmark results can be found here [model_quantization_results.py](https://sony.github.io/model_optimization/benchmark/results/model_quantization_results.py) 


## License
This project is licensed under [Apache License 2.0](https://sony.github.io/model_optimization/LICENSE.md).
However, please note that different external packages have their own licenses. When using this project, you have the option to choose one of the following external packages:

- torchvision: [license link](https://github.com/UiPath/torchvision/blob/master/LICENSE), [license copy](https://sony.github.io/model_optimization/benchmark/pytorch_code/torchvision/LICENSE)
- timm: [license link](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE), [license copy](https://sony.github.io/model_optimization/benchmark/pytorch_code/timm/LICENSE)
- ultralytics: [license link](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), [license copy](https://sony.github.io/model_optimization/benchmark/pytorch_code/ultralytics/LICENSE)
