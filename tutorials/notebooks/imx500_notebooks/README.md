# Sony-IMX500 Notebooks

Here we provide examples on quantizing pre-trained models for deployment on Sony-IMX500 processing platform.
We will cover various tasks and demonstrate the necessary steps to achieve efficient quantization for optimal
deployment performance.

*<sup>[1]</sup>* <sub>Jupyter notebook explaining how to generate IMX500 compatible model.</sub>

*<sup>[2]</sup>* <sub>Floating point model with necessary adjustments for MCT compatibility. If none, the source model is compatible with MCT.</sub>

*<sup>[3]</sup>* <sub>Expected model accuracy on IMX500.</sub>

<table>
    <tr>
        <th rowspan="1">Task</th>
        <th rowspan="1">Model Name</th>
        <th rowspan="1">Notebook<sup>[1]</sup></th>
        <th rowspan="1">Source Repository</th>
        <th rowspan="1">Adjusted Model<sup>[2]</sup></th>
        <th rowspan="1">Dataset Name</th>
        <th rowspan="1">Float Model Accuracy</th>
        <th rowspan="1">Compressed Model Accuracy<sup>[3]</sup></th>
    </tr>
    <!-- Classification Models (ImageNet) -->
    <tr>
        <td rowspan="9">Classification</td>
        <td>MobilenetV2</td>
        <td> <a href="keras/example_keras_mobilenetv2_for_imx500.ipynb">ipynb (Keras)</a></td>
        <td><a href="https://keras.io/api/applications/mobilenet/">Keras Applications</a></td>
        <td></td>
        <td>ImageNet</td>
        <td>71.85</td>
        <td>71.67</td>
    </tr>
    <tr>
        <td>MobileVit</td>
        <td> <a href="pytorch/pytorch_mobilevit_xs_for_imx500.ipynb">ipynb (PyTorch)</a></td>
        <td><a href="https://github.com/huggingface/pytorch-image-models">Timm</a></td>
        <td><a href="https://huggingface.co/SSI-DNN/pytorch_mobilevit_xs">mct-model-garden</a></td>
        <td>ImageNet</td>
        <td>74.64</td>
        <td>72.56</td>
    </tr>
    <tr>
        <td>regnety_002.pycls_in1k</td>
        <td rowspan="3"> <a href="pytorch/pytorch_timm_classification_model_for_imx500.ipynb">ipynb (PyTorch)</a></td>
        <td><a href="https://github.com/huggingface/pytorch-image-models">Timm</a></td>
        <td></td>
        <td>ImageNet</td>
        <td>70.28</td>
        <td>69.9</td>
    </tr>
    <tr>
        <td>regnetx_002.pycls_in1k</td>
        <td><a href="https://github.com/huggingface/pytorch-image-models">Timm</a></td>
        <td></td>
        <td>ImageNet</td>
        <td>68.752</td>
        <td>68.652</td>
    </tr>
    <tr>
        <td>regnety_004.pycls_in1k</td>
        <td><a href="https://github.com/huggingface/pytorch-image-models">Timm</a></td>
        <td></td>
        <td>ImageNet</td>
        <td>74.026</td>
        <td>73.72</td>
    </tr>
    <tr>
        <td>mnasnet1_0</td>
        <td rowspan="4"> <a href="pytorch/pytorch_torchvision_classification_model_for_imx500.ipynb">ipynb (PyTorch)</a></td>
        <td><a href="https://pytorch.org/vision/stable/models/generated/torchvision.models.mnasnet1_0.html#torchvision.models.MNASNet1_0_Weights">torchvision</a></td>
        <td></td>
        <td>ImageNet</td>
        <td>73.47</td>
        <td>73.16</td>
    </tr>
    <tr>
        <td>mobilenet_v2</td>
        <td><a href="https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html#torchvision.models.MobileNet_V2_Weights">torchvision</a></td>
        <td></td>
        <td>ImageNet</td>
        <td>72.01</td>
        <td>71.25</td>
    </tr>
    <tr>
        <td>regnet_y_400mf</td>
        <td><a href="https://pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_400mf.html#torchvision.models.RegNet_Y_400MF_Weights">torchvision</a></td>
        <td></td>
        <td>ImageNet</td>
        <td>74.03</td>
        <td>73.69</td>
    </tr>
    <tr>
        <td>shufflenet_v2_x1_5</td>
        <td><a href="https://pytorch.org/vision/stable/models/generated/torchvision.models.shufflenet_v2_x1_5.html#torchvision.models.ShuffleNet_V2_X1_5_Weights">torchvision</a></td>
        <td></td>
        <td>ImageNet</td>
        <td>69.34</td>
        <td>69.04</td>
    </tr>
    <!-- Object Detection Models (COCO) -->
    <tr>
        <td rowspan="4">Object Detection</td>
        <td>YOLOv8n</td>
        <td> <a href="keras/keras_yolov8n_for_imx500.ipynb">ipynb (Keras)</a></td>
        <td><a href="https://github.com/ultralytics">Ultralytics</a></td>
        <td><a href="https://huggingface.co/SSI-DNN/keras_yolov8n_640x640_pp">mct-model-garden</a></td>
        <td>COCO</td>
        <td>37.3</td>
        <td>35.1</td>
    </tr>
    <tr>
        <td>YOLOv8n</td>
        <td> <a href="pytorch/pytorch_yolov8n_for_imx500.ipynb">ipynb (PyTorch)</a></td>
        <td><a href="https://github.com/ultralytics">Ultralytics</a></td>
        <td><a href="https://huggingface.co/SSI-DNN/pytorch_yolov8n_640x640_bb_decoding">mct-model-garden</a></td>
        <td>COCO</td>
        <td>37.3</td>
        <td>35.1</td>
    </tr>
    <tr>
        <td>NanoDet-Plus-m-416</td>
        <td> <a href="keras/example_keras_nanodet_plus_for_imx500.ipynb">ipynb (Keras)</a></td>
        <td><a href="https://github.com/RangiLyu/nanodet">Nanodet</a></td>
        <td><a href="https://huggingface.co/SSI-DNN/keras_nanodet_plus_x1.5_416x416">mct-model-garden</a></td>
        <td>COCO</td>
        <td>34.1</td>
        <td>32.2</td>
    </tr>
    <tr>
        <td>EfficientDet-lite0</td>
        <td> <a href="keras/example_keras_effdet_lite0_for_imx500.ipynb">ipynb (Keras)</a></td>
        <td> <a href="https://github.com/rwightman/efficientdet-pytorch">efficientdet-pytorch</a></td>
        <td><a href="https://github.com/sony/model_optimization/blob/main/tutorials/mct_model_garden/models_keras/efficientdet/effdet_keras.py">mct-model-garden</a></td>
        <td>COCO</td>
        <td>27.0</td>
        <td>25.2</td>
    </tr>
    <tr>
        <td>Semantic Segmentation</td>
        <td>Deeplabv3plus</td>
        <td> <a href="keras/keras_deeplabv3plus_for_imx500.ipynb">ipynb (Keras)</a></td>
        <td> <a href="https://github.com/bonlime/keras-deeplab-v3-plus">bonlime</a></td>
        <td><a href="https://huggingface.co/SSI-DNN/keras_deeplabv3_plus_320">mct-model-garden</a></td>
        <td>PASCAL VOC</td>        
        <td>76.935</td>
        <td>76.778</td>
    </tr>
    <tr>
        <td >Instance Segmentation</td>
        <td>YOLOv8n-seg</td>
        <td> <a href="pytorch/pytorch_yolov8n_seg_for_imx500.ipynb">ipynb (PyTorch)</a></td>
        <td><a href="https://github.com/ultralytics">Ultralytics</a></td>
        <td><a href="https://huggingface.co/SSI-DNN/pytorch_yolov8n_inst_seg_640x640">mct-model-garden</a></td>
        <td>COCO</td>        
        <td>30.5</td>
        <td>29.5</td>
    </tr>
    <tr>
        <td>Pose Estimation</td>
        <td>YOLOv8n-pose</td>
        <td> <a href="pytorch/pytorch_yolov8n_pose_for_imx500.ipynb">ipynb (PyTorch)</a></td>
        <td><a href="https://github.com/ultralytics">Ultralytics</a></td>
        <td><a href="https://huggingface.co/SSI-DNN/pytorch_yolov8n_640x640">mct-model-garden</a></td>
        <td>COCO</td>
        <td>50.4</td>
        <td>47.1</td>
    </tr>

</table>

