# Sony-IMX500 Notebooks

Here we provide examples on quantizing pre-trained models for deployment on Sony-IMX500 processing platform.
We will cover various tasks and demonstrate the necessary steps to achieve efficient quantization for optimal
deployment performance.

<table>
    <tr>
        <th rowspan="1">Task</th>
        <th rowspan="1">Model Name</th>
        <th rowspan="1">Framework</th>
        <th rowspan="1">Dataset Name</th>
    </tr>
    <!-- Classification Models (ImageNet) -->
    <tr>
        <td rowspan="2">Classification</td>
        <td> <a href="keras/ptq/example_keras_imagenet.ipynb">mobilenet_v2</a></td>
        <td>Keras</td>
        <td>ImageNet</td>
    </tr>
    <tr>
        <td> <a href="pytorch/ptq/pytorch_mobilevit_xs_for_imx500.ipynb">MobileVit</a></td>
        <td>PyTorch</td>
        <td>ImageNet</td>
    </tr>
    <!-- Object Detection Models (COCO) -->
    <tr>
        <td rowspan="4">Object Detection</td>
        <td> <a href="keras/ptq/keras_yolov8n_for_imx500.ipynb">yolov8n</a></td>
        <td>Keras</td>
        <td>COCO</td>
    </tr>
    <tr>
        <td> <a href="pytorch/ptq/keras_yolov8n_for_imx500.ipynb">yolov8n</a></td>
        <td>PyTorch</td>
        <td>COCO</td>
    </tr>
    <tr>
        <td> <a href="keras/ptq/example_keras_nanodet_plus.ipynb">NanoDet-Plus-m-416</a></td>
        <td>Keras</td>
        <td>COCO</td>
    </tr>
    <tr>
        <td> <a href="keras/ptq/example_keras_effdet_lite0.ipynb">EfficientDet-lite0</a></td>
        <td>Keras</td>
        <td>COCO</td>
    </tr>
    <tr>
        <td>Semantic Segmentation</td>
        <td> <a href="keras/ptq/keras_deeplabv3plus_for_imx500.ipynb">Deeplabv3plus</a></td>
        <td>Keras</td>
        <td>PASCAL VOC</td>
    </tr>
    <tr>
        <td >Instance Segmentation</td>
        <td>yolov8n-seg</td>
        <td>PyTorch</td>
        <td>COCO</td>
    </tr>
    <tr>
        <td>Pose Estimation</td>
        <td>yolov8n-pose</td>
        <td>PyTorch</td>
        <td>COCO</td>
    </tr>

</table>

