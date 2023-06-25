# MCT Quick Start - Ultralytics Example 

This example demonstrates how to use MCT 'quick start' with Ultralytics object detection model.


## Getting Started

### Installation 
Install the required library using `pip`:
```bash
pip install ultralytics==8.0.47
 ```

### Usage Examples

In the following example, we are evaluating the MCT on pre-trained yolov8n, taken from the Ultralytics models library.

Note that we assume the command is run from the command line and that the [main.py](../../main.py) script is located in the current directory.
Additionally, it is expected that the [model_optimization](../../../../) folder is included in python path.
```csh 
setenv PYTHONPATH /path/to/model_optimization/folder
```

```csh 
python main.py --model_name yolov8n --model_library ultralytics --representative_dataset_folder ../datasets/coco/images/train2017 --num_representative_images 64 --batch_size 1 
```
In this example, we are running the main.py script with the following parameters:
- `--model_name`: Specifies the name of the model to be used out of Ultralytics models (yolov8n in this case).
- `--model_library`: Specifies the package of the pre-trained models from which the model name is taken (in this case, ultralytics).
- `--representative_dataset_folder`: Specifies the path to the local copy of the dataset to be used for quantization. In this case, we use the 'train' split of the downloaded dataset provided by Ultralytics.
- `--num_representative_images`: Specifies the number of representative images to be used for quantization.
- `--batch_size`: Specifies the batch size to be used.

Please note that during the first model evaluation, Ultralytics downloads the COCO dataset to the folder specified in the 'coco.yaml' file. By default, the dataset is downloaded to '../datasets/coco'. Therefore, the 'validation_dataset_folder' value is not required in this case.

For the representative dataset, it is expected to follow the same format as the downloaded COCO dataset. For instance, you can use the 'train' split of the COCO dataset as the representative dataset or create a new split with the same format.


## Model Replacers
During the process, we perform a few manipulations to achieve better quantization results:

1. We replace certain modules with modules supported by `torch.fx`, which our project relies on. The `torch.fx` toolkit helps us acquire a static graph representation from PyTorch models, enabling model compression manipulations like batch norm folding.
2. We remove the last part of the detection head, responsible for bounding box decoding, and include it as part of the postprocessing. You can find the additional postprocessing in the new definition of the `postprocess` method under [replacer.py](./replacers.py).


## License
This project is licensed under [Apache License 2.0](../../../../LICENSE.md).
Ultralytics has the following license requirements: 
- ultralytics: [license link](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), [license copy](./LICENSE)
