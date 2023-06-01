# MCT Benchmark 

This module provides a tool for testing and benchmarking the performance of MCT.
It is designed to assist developers in evaluating the compression efficiency on a large-scale set of pre-trained models.


## Features
- Testing: Performance of compressed models against uncompressed models
- Compression/Quantization: Several quantization techniques are available, based on MCT features set.  
- Models: Support popular pre-trained models available online


## Getting Started
### Installation 
- Install MCT - Please refer to [MCT readme](https://github.com/sony/model_optimization/blob/main/README.md)
- Install one of the following package of pre-trained models

### Supported packages of pre-trained models
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [timm](https://timm.fast.ai)
- [ultralytics](https://ultralytics.com)

### Usage Example
In this example, we are evaluating the MCT on pre-trained mobilenet_v2, taken from torchvision models library
```python
python main.py --model_name mobilenet_v2 --model_library torchvision --validation_dataset_folder <my path> --representative_dataset_folder <my path> 
```
Please note that the above example assumes the command is run from the command line and that the [main.py](https://sony.github.io/model_optimization/benchmark/main.py) script is in the current directory

## Results
The latest MCT benchmark results can be found here [model_quantization_results.py](https://sony.github.io/model_optimization/benchmark/results/model_quantization_results.py) 

## Contributions

## License
[Apache License 2.0](LICENSE.md).
