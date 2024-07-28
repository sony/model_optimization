# Sony-IMX500 Notebooks

Here we provide examples on quantizing pre-trained models for deployment on Sony-IMX500 processing platform.
We will cover various tasks and demonstrate the necessary steps to achieve efficient quantization for optimal
deployment performance.

<table>
    <tr>
        <th rowspan="1">Task</th>
        <th rowspan="1">Model Name</th>
        <th rowspan="1">Framework</th>
        <th rowspan="1">Source Repository</th>
        <th rowspan="1">Dataset Name</th>
        <th rowspan="1">Float Model Accuracy</th>
        <th rowspan="1">Compressed Model Accuracy</th>
    </tr>
    <!-- Classification Models (ImageNet) -->
    <tr>
        <td rowspan="5">Classification</td>
        <td> <a href="keras/example_keras_mobilenetv2_for_imx500.ipynb">MobilenetV2</a></td>
        <td>Keras</td>
        <td>Keras Applications</td>
        <td>ImageNet</td>
        <td>71.85</td>
        <td>71.67</td>
    </tr>
    <tr>
        <td> <a href="pytorch/pytorch_mobilevit_xs_for_imx500.ipynb">MobileVit</a></td>
        <td>PyTorch</td>
        <td>MCT Model Garden</td>
        <td>ImageNet</td>
        <td>74.64</td>
        <td>72.56</td>
    </tr>
    <tr>
        <td> <a href="pytorch/pytorch_timm_classification_model_for_imx500.ipynb">regnety_002.pycls_in1k</a></td>
        <td>PyTorch</td>
        <td>Timm</td>
        <td>ImageNet</td>
        <td>70.28</td>
        <td>69.9</td>
    </tr>
    <tr>
        <td> <a href="pytorch/pytorch_timm_classification_model_for_imx500.ipynb">regnetx_002.pycls_in1k</a></td>
        <td>PyTorch</td>
        <td>Timm</td>
        <td>ImageNet</td>
        <td>68.752</td>
        <td>68.652</td>
    </tr>
    <tr>
        <td> <a href="pytorch/pytorch_timm_classification_model_for_imx500.ipynb">regnety_004.pycls_in1k</a></td>
        <td>PyTorch</td>
        <td>Timm</td>
        <td>ImageNet</td>
        <td>74.026</td>
        <td>73.72</td>
    </tr>
    <!-- Object Detection Models (COCO) -->
    <tr>
        <td rowspan="4">Object Detection</td>
        <td> <a href="keras/keras_yolov8n_for_imx500.ipynb">YOLOv8n</a></td>
        <td>Keras</td>
        <td>MCT Model Garden</td>
        <td>COCO</td>
        <td>37.3</td>
        <td>35.1</td>
    </tr>
    <tr>
        <td> <a href="pytorch/pytorch_yolov8n_for_imx500.ipynb">YOLOv8n</a></td>
        <td>PyTorch</td>
        <td>MCT Model Garden</td>
        <td>COCO</td>
        <td>37.3</td>
        <td>35.1</td>
    </tr>
    <tr>
        <td> <a href="keras/example_keras_nanodet_plus_for_imx500.ipynb">NanoDet-Plus-m-416</a></td>
        <td>Keras</td>
        <td>MCT Model Garden</td>
        <td>COCO</td>
        <td>34.1</td>
        <td>32.2</td>
    </tr>
    <tr>
        <td> <a href="keras/example_keras_effdet_lite0_for_imx500.ipynb">EfficientDet-lite0</a></td>
        <td>Keras</td>
        <td>MCT Model Garden</td>
        <td>COCO</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Semantic Segmentation</td>
        <td> <a href="keras/keras_deeplabv3plus_for_imx500.ipynb">Deeplabv3plus</a></td>
        <td>Keras</td>
        <td>MCT Model Garden</td>
        <td>PASCAL VOC</td>        
        <td>76.935</td>
        <td>76.778</td>
    </tr>
    <tr>
        <td >Instance Segmentation</td>
        <td> <a href="pytorch/pytorch_yolov8n_seg_for_imx500.ipynb">YOLOv8n-seg</a></td>
        <td>PyTorch</td>
        <td>MCT Model Garden</td>
        <td>COCO</td>        
        <td>30.5</td>
        <td>29.5</td>
    </tr>
    <tr>
        <td>Pose Estimation</td>
        <td> <a href="pytorch/pytorch_yolov8n_pose_for_imx500.ipynb">YOLOv8n-pose</a></td>
        <td>PyTorch</td>
        <td>MCT Model Garden</td>
        <td>COCO</td>
        <td>50.4</td>
        <td>47.1</td>
    </tr>

</table>

