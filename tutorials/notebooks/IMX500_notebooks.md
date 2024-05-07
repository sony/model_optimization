# Sony-IMX500 Notebooks

Here we provide examples on quantizing pre-trained models for deployment on Sony-IMX500 processing platform.
We will cover various tasks and demonstrate the necessary steps to achieve efficient quantization for optimal
deployment performance.

<table>
    <tr>
        <th rowspan="2">Task</th>
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
        <td rowspan="2">Classification</td>
        <td> <a href="keras/ptq/example_keras_imagenet.ipynb">mobilenet_v2</a></td>
        <td>ImageNet</td>
        <td>72.15%</td>
        <td>13.88</td>
        <td>71.88%</td>
        <td>3.47</td>
    </tr>
    <tr>
        <td>MobileVit</td>
        <td>ImageNet</td>
        <td>72.78%</td>
        <td>17.24</td>
        <td>67.42%</td>
        <td>4.31</td>
    </tr>
    <!-- Object Detection Models (COCO) -->
    <tr>
        <th colspan="8">mAP</th>
    </tr>
    <tr>
        <td rowspan="3">Object Detection</td>
        <td> <a href="keras/ptq/keras_yolov8n_for_imx500.ipynb">yolov8n</a></td>
        <td>COCO</td>
        <td>37.04</td>
        <td>12.6</td>
        <td>35.0</td>
        <td>3.15</td>
    </tr>
    <tr>
        <td>Nanodet</td>
        <td>COCO</td>
        <td>34.1</td>
        <td>8.0</td>
        <td>32.1</td>
        <td>2.0</td>
    </tr>
    <tr>
        <td>Efficientdet-lite0</td>
        <td>COCO</td>
        <td>25.1</td>
        <td>8.0</td>
        <td>24.1</td>
        <td>2.0</td>
    </tr>
    <tr>
        <td>Semantic Segmentation</td>
        <td>Deeplab-V3</td>
        <td>COCO</td>
        <td>50.92</td>
        <td>13.12</td>
        <td>49.18</td>
        <td>3.28</td>
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
    <tr>
        <td>Pose Estimation</td>
        <td>yolov8n-pose</td>
        <td>COCO</td>
        <td>50.92</td>
        <td>13.12</td>
        <td>49.18</td>
        <td>3.28</td>
    </tr>

</table>

