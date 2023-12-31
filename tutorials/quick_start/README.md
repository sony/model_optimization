# MCT Quick Start 

This example project demonstrates the capabilities of MCT (Model Compression Toolkit) and illustrates its interface
with various model collections libraries. By accessing a wide range of pre-trained models, this project allows users to
generate a quantized version of their chosen model with a single click. 

Currently, the project supports a selection of models from each library. However, our ongoing goal is to continually
expand the support, aiming to include more models
in the future.   

## Supported Features
 - Quantize pretrained models from various model [libraries](#supported-libraries)
 - Evaluate the performance of the floating point model and the quantized model.
 - Use MCT's Post-Training Quantization (PTQ) scheme [link to example](#basic-model-quantization-example---post-training-quantization-ptq)
 - Use MCT's advanced quantization techniques (such as GPTQ and Mixed Precision) [link to the example](#advanced-model-quantization-example)


## Results

<table>
    <tr>
        <th rowspan="2">Task</th>
        <th rowspan="2">Model Source</th>
        <th rowspan="2">Model Name</th>
        <th rowspan="2">Dataset Name</th>
        <th colspan="2">Float</th>
        <th colspan="2">Quantized</th>
    </tr>
    <tr>
        <th>Performance</th>
        <th>Size [MB]</th>
        <th>Performance</th>
        <th>Size [MB]</th>
    </tr>
    <!-- Classification Models (ImageNet) -->
    <tr>
        <th colspan="8">Top-1 Accuracy</th>
    </tr>
    <tr>
        <td rowspan="12">Classification</td>
        <td rowspan="4"><a href="https://github.com/pytorch/vision">torchvision</a></td>
        <td> <a href="https://colab.research.google.com/github/sony/model_optimization/tutorials/notebooks/example_quick_start_imagenet.ipynb">mobilenet_v2</a></td>
        <td>ImageNet</td>
        <td>72.15%</td>
        <td>13.88</td>
        <td>71.88%</td>
        <td>3.47</td>
    </tr>
    <tr>
        <td>regnet_y_400mf</td>
        <td>ImageNet</td>
        <td>75.78%</td>
        <td>17.24</td>
        <td>75.42%</td>
        <td>4.31</td>
    </tr>
    <tr>
        <td>shufflenet_v2_x0_5</td>
        <td>ImageNet</td>
        <td>60.55%</td>
        <td>5.44</td>
        <td>59.7%</td>
        <td>1.36</td>
    </tr>
    <tr>
        <td>squeezenet1_0</td>
        <td>ImageNet</td>
        <td>58.1%</td>
        <td>4.96</td>
        <td>57.67%</td>
        <td>1.24</td>
    </tr>
    <tr>
        <td rowspan="5"><a href="https://github.com/rwightman/pytorch-image-models">timm</a></td>
        <td>regnetx_002</td>
        <td>ImageNet</td>
        <td>68.76%</td>
        <td>10.68</td>
        <td>68.27%</td>
        <td>2.67</td>
    </tr>
    <tr>
        <td>regnety_008</td>
        <td>ImageNet</td>
        <td>76.32%</td>
        <td>24.92</td>
        <td>75.98%</td>
        <td>6.23</td>
    </tr>
    <tr>
        <td>resnet10t</td>
        <td>ImageNet</td>
        <td>66.56%</td>
        <td>21.72</td>
        <td>66.43%</td>
        <td>5.43</td>
    </tr>
    <tr>
        <td>resnet18</td>
        <td>ImageNet</td>
        <td>69.76%</td>
        <td>46.72</td>
        <td>69.61%</td>
        <td>11.68</td>
    </tr>
    <tr>
        <td>efficientnet_es</td>
        <td>ImageNet</td>
        <td>78.08%</td>
        <td>21.56</td>
        <td>77.74%</td>
        <td>5.39</td>
    </tr> 
    <tr>
        <td rowspan="3"><a href="https://github.com/keras-team/keras-applications">keras_applications</a></td>
        <td>mobilenet_v2.MobileNetV2</td>
        <td>ImageNet</td>
        <td>71.85%</td>
        <td>13.88</td>
        <td>71.57%</td>
        <td>3.47</td>
    </tr>
    <tr>
        <td>efficientnet_v2.EfficientNetV2B0</td>
        <td>ImageNet</td>
        <td>78.41%</td>
        <td>28.24</td>
        <td>77.44%</td>
        <td>7.06</td>
    </tr>
    <tr>
        <td>resnet50.ResNet50</td>
        <td>ImageNet</td>
        <td>74.22%</td>
        <td>102</td>
        <td>74.08%</td>
        <td>25.5</td>
    </tr>
    <!-- Object Detection Models (COCO) -->
    <tr>
        <th colspan="8">mAP</th>
    </tr>
    <tr>
        <td rowspan="2">Object Detection</td>
        <td rowspan="3"><a href="https://github.com/ultralytics">ultralytics</a></td>
        <td>yolov8n</td>
        <td>COCO</td>
        <td>37.04</td>
        <td>12.6</td>
        <td>36.17</td>
        <td>3.15</td>
    </tr>
    <tr>
        <td>yolov8m</td>
        <td>COCO</td>
        <td>49.99</td>
        <td>103.6</td>
        <td>49.4</td>
        <td>25.9</td>
    </tr>
    <tr>
        <td >Instance Segmentation</td>
        <td>yolov8n-seg</td>
        <td>COCO</td>
        <td>30.51</td>
        <td>13.6</td>
        <td>30.18</td>
        <td>3.4</td>
    </tr>
</table>

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
More details and examples for using Ultrlytics models can be found in this [readme](./pytorch_fw/ultralytics_lib/README.md)   

## External Package Versions

The following external packages were tested with this project:

- torchvision: Version 0.14.0
- timm: Version 0.6.13
- ultralytics: Version 8.0.47
- keras-applications: Version 2.9.0

## License
This project is licensed under [Apache License 2.0](../../LICENSE.md).
However, please note that different external packages have their own licenses. When using this project, you have the option to choose one of the following external packages:

- torchvision: [license link](https://github.com/UiPath/torchvision/blob/master/LICENSE), [license copy](./pytorch_fw/torchvision_lib/LICENSE)
- timm: [license link](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE), [license copy](./pytorch_fw/timm_lib/LICENSE)
- ultralytics: [license link](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), [license copy](./pytorch_fw/ultralytics_lib/LICENSE)
- keras-applications: [license link](https://github.com/keras-team/keras-applications/blob/master/LICENSE), [license copy](./keras_fw/keras_applications/LICENSE)
