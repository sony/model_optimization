# Sony-IMX500 Notebooks

Here we provide examples on quantizing pre-trained models for deployment on Sony-IMX500 processing platform.
We will cover various tasks and demonstrate the necessary steps to achieve efficient quantization for optimal
deployment performance.
  
  | Task                                                            | Model          | Source Repository                                                   | Notebook                                                                                                |
  |-----------------------------------------------------------------|----------------|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
  | Classification                                                  | MobileNetV2    | [Keras Applications](https://keras.io/api/applications/)            | [Keras notebook](keras/ptq/example_keras_imagenet.ipynb)         |
  | Object Detection                                                | YOLOv8n        | [Ultralytics](https://github.com/ultralytics/ultralytics)           | [Keras notebook](keras/ptq/keras_yolov8n_for_imx500.ipynb)       | 
  | Semantic Segmentation                                           | DeepLabV3-Plus | [bonlime's repo](https://github.com/bonlime/keras-deeplab-v3-plus)  | [Keras notebook](keras/ptq/keras_deeplabv3plus_for_imx500.ipynb) |

