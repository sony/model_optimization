# MCT Model Garden 

MCT Model Garden is a collection of models sourced from various repositories and adjusted for quantization using Model Compression Toolkit (MCT).

Adjustments are sometimes necessary before applying MCT due to various reasons, such as:
- Enabling the conversion of the model to a static graph (the initial stage in MCT).
- Enhancing the quantization outcome.
- Converting unsupported operators.

Note that in many cases, adjustments are unnecessary before applying MCT.

## Models

| Model        | Source Repository           | MCT Model Garden                                                                                                        | 
|--------------|-----------------------------|-------------------------------------------------------------------------------------------------------------------------|
| EfficientDet | [efficientdet-pytorch](https://github.com/rwightman/efficientdet-pytorch) | [model](https://github.com/sony/model_optimization/tree/main/tutorials/mct_model_garden/models_keras/efficientdet) |
| Nanodet-Plus | [Nanodet-Plus](https://github.com/RangiLyu/nanodet)         | [model](https://github.com/sony/model_optimization/tree/main/tutorials/mct_model_garden/models_keras/nanodet)      |                                                                                                                
| Yolov8n      | [Ultralytics](https://github.com/ultralytics/ultralytics)          | [model](https://github.com/sony/model_optimization/tree/main/tutorials/mct_model_garden/models_keras/yolov8)       |



## License
This project is licensed under [Apache License 2.0](../../LICENSE.md).
However, please note that different repositories have their own licenses. Therefore, when using a model from 
this library, it's essential to also comply with the licensing terms of the source repositories.
