# MCT Benchmark 

This module provides a tool for testing and benchmarking the performance of MCT.
It is designed to assist developers in evaluating the compression efficiency and the accuracy preservation on a large-scale set of pre-trained models.


## Features
- Testing:  The module allows to test the performance of compressed models against uncompressed models
- Model compression and quantization capabilities: Several techniques are available, based on MCT features set.  
- Large-scale set of models: Supports popular pre-trained model's libraries which are available online


## Getting Started
### Installation 
- Install MCT - Please refer to [MCT readme](https://github.com/sony/model_optimization/blob/main/README.md)
- Install one of the following package of pre-trained models

### Supported packages of pre-trained models
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [timm](https://timm.fast.ai)
- [ultralytics](https://ultralytics.com)

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

Please note that the above example assumes the command is run from the command line and that the [main.py](https://sony.github.io/model_optimization/benchmark/main.py) script is in the current directory

Make sure to refer to the script itself to understand all available parameters and their usage.
## Results
The latest MCT benchmark results can be found here [model_quantization_results.py](https://sony.github.io/model_optimization/benchmark/results/model_quantization_results.py) 

## Contributions
This contribution includes an evaluation and benchmarking test for the MCT project. 
The benchmarking code utilizes the following external libraries: 
[torchvision](https://pytorch.org/vision/stable/index.html), 
[timm](https://github.com/huggingface/pytorch-image-models),
and [ultralytics](https://github.com/ultralytics/ultralytics).

We extend our gratitude to the authors and contributors of these libraries for their valuable work.

## License
MCT license:
[Model CompressionToolkit](https://github.com/sony/model_optimization/blob/main/LICENSE.md)

External Libraries Used:
- [torchvision](https://github.com/UiPath/torchvision/blob/master/LICENSE)
- [timm](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE)
- [ultralytics](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
