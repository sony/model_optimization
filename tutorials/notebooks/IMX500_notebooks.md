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
        <td>MobileVit</td>
        <td>PyTorch</td>
        <td>ImageNet</td>
    </tr>
    <!-- Object Detection Models (COCO) -->
    <tr>
        <td rowspan="3">Object Detection</td>
        <td> <a href="keras/ptq/keras_yolov8n_for_imx500.ipynb">yolov8n</a></td>
        <td>Keras</td>
        <td>COCO</td>
    </tr>
    <tr>
        <td>Nanodet</td>
        <td>Keras</td>
        <td>COCO</td>
    </tr>
    <tr>
        <td>Efficientdet-lite0</td>
        <td>Keras</td>
        <td>COCO</td>
    </tr>
    <tr>
        <td>Semantic Segmentation</td>
        <td>deeplabv3-plus</td>
        <td>Keras</td>
        <td>COCO</td>
    </tr>
    <tr>
        <td >Instance Segmentation</td>
        <td>yolov8n-seg</td>
        <td>Keras</td>
        <td>COCO</td>
    </tr>
    <tr>
        <td>Pose Estimation</td>
        <td>yolov8n-pose</td>
        <td>PyTorch</td>
        <td>COCO</td>
    </tr>

</table>

