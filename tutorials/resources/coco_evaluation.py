# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import json
import cv2
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco80_to_coco91(x):
    """
    Converts COCO 80-class indices to COCO 91-class indices.

    Args:
        x (numpy.ndarray): An array of COCO 80-class indices.

    Returns:
        numpy.ndarray: An array of corresponding COCO 91-class indices.
    """
    coco91Indexs = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
         63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90])
    return coco91Indexs[x.astype(np.int32)]


def clip_boxes(boxes, h, w):
    """
    Clip bounding boxes to stay within the image boundaries.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes in format [y_min, x_min, y_max, x_max].
        h (int): Height of the image.
        w (int): Width of the image.

    Returns:
        numpy.ndarray: Clipped bounding boxes.
    """
    boxes[..., 0] = np.clip(boxes[..., 0], a_min=0, a_max=h)
    boxes[..., 1] = np.clip(boxes[..., 1], a_min=0, a_max=w)
    boxes[..., 2] = np.clip(boxes[..., 2], a_min=0, a_max=h)
    boxes[..., 3] = np.clip(boxes[..., 3], a_min=0, a_max=w)
    return boxes


def scale_boxes(boxes, H, W, h_image, w_image):
    """
    Scale and offset bounding boxes based on model output size and original image size.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes in format [y_min, x_min, y_max, x_max].
        H (int): Model output height.
        W (int): Model output width.
        h_image (int): Original image height.
        w_image (int): Original image width.

    Returns:
        numpy.ndarray: Scaled and offset bounding boxes.
    """
    deltaH, deltaW = 0, 0
    scale_H, scale_W = h_image / H, w_image / W

    # Scale and offset boxes
    boxes[..., 0] = (boxes[..., 0] - deltaH) * scale_H
    boxes[..., 1] = (boxes[..., 1] - deltaW) * scale_W
    boxes[..., 2] = (boxes[..., 2] - deltaH) * scale_H
    boxes[..., 3] = (boxes[..., 3] - deltaW) * scale_W

    # Clip boxes
    boxes = clip_boxes(boxes, h_image, w_image)
    return boxes


def format_results(outputs, img_ids, orig_img_dims):
    """
    Format model outputs into a list of detection dictionaries.

    Args:
        outputs (list): List of model outputs, typically containing bounding boxes, scores, and labels.
        img_ids (list): List of image IDs corresponding to each output.
        orig_img_dims (list): List of tuples representing the original image dimensions for each output.

    Returns:
        list: A list of detection dictionaries, each containing information about the detected object.
    """
    detections = []

    # Process model outputs and convert to detection format
    for idx, output in enumerate(outputs):
        image_id = img_ids[idx]
        scores = output[1].numpy().squeeze() # Extract scores
        labels = (coco80_to_coco91(
            output[2].numpy())).squeeze()  # Convert COCO 80-class indices to COCO 91-class indices
        boxes = output[0].numpy().squeeze() # Extract bounding boxes
        boxes = scale_boxes(boxes, 1, 1, orig_img_dims[idx][0], orig_img_dims[idx][1])

        for score, label, box in zip(scores, labels, boxes):
            detection = {
                "image_id": image_id,
                "category_id": label,
                "bbox": [box[1], box[0], box[3] - box[1], box[2] - box[0]],
                "score": score
            }
            detections.append(detection)

    return detections

# COCO evaluation class
class CocoEval:
    def __init__(self, path2json):
        """
        Initialize the CocoEval class.

        Args:
            path2json (str): Path to the COCO JSON file containing ground truth annotations.
        """
        # Load ground truth annotations
        self.coco_gt = COCO(path2json)

        # A list of reformatted model outputs
        self.all_detections = []

    def add_batch_detections(self, outputs, targets):
        """
        Add batch detections to the evaluation.

        Args:
            outputs (list): List of model outputs, typically containing bounding boxes, scores, and labels.
            targets (list): List of ground truth annotations for the batch.
        """
        img_ids, _outs = [], []
        orig_img_dims = []
        for idx, t in enumerate(targets):
            if len(t) > 0:
                img_ids.append(t[0]['image_id'])
                orig_img_dims.append(t[0]['orig_img_dims'])
                _outs.append([outputs[0][idx], outputs[1][idx], outputs[2][idx], outputs[3][idx]])

        batch_detections = format_results(_outs, img_ids, orig_img_dims)

        self.all_detections.extend(batch_detections)

    def result(self):
        """
        Calculate and print evaluation results.

        Returns:
            list: COCO evaluation statistics.
        """
        # Initialize COCO evaluation object
        self.coco_dt = self.coco_gt.loadRes(self.all_detections)
        coco_eval = COCOeval(self.coco_gt, self.coco_dt, 'bbox')

        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Print mAP results
        print("mAP: {:.4f}".format(coco_eval.stats[0]))

        return coco_eval.stats

    def reset(self):
        """
        Reset the list of detections to prepare for a new evaluation.
        """
        self.all_detections = []

def load_and_preprocess_image(image_path, preprocess):
    """
    Load and preprocess an image from a given file path.

    Args:
        image_path (str): Path to the image file.
        preprocess (function): Preprocessing function to apply to the loaded image.

    Returns:
        numpy.ndarray: Preprocessed image.
    """
    image = cv2.imread(image_path)
    image = preprocess(image)
    return image

def coco_dataset_generator(dataset_folder, annotation_file, preprocess, batch_size=1):
    """
    Generator function for loading and preprocessing images and their annotations from a COCO-style dataset.

    Args:
        dataset_folder (str): Path to the dataset folder containing image files.
        annotation_file (str): Path to the COCO-style annotation JSON file.
        preprocess (function): Preprocessing function to apply to each loaded image.
        batch_size (int): The desired batch size.

    Yields:
        Tuple[numpy.ndarray, list]: A tuple containing a batch of images (as a NumPy array) and a list of annotations
        for each image in the batch.
    """
    # Load COCO annotations from a JSON file (e.g., 'annotations.json')
    with open(annotation_file, 'r') as f:
        coco_annotations = json.load(f)

    # Initialize a dictionary to store annotations grouped by image ID
    annotations_by_image = {}

    # Iterate through the annotations and group them by image ID
    for annotation in coco_annotations['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)

    # Initialize a list to collect images and annotations for the current batch
    batch_images = []
    batch_annotations = []
    total_images = len(coco_annotations['images'])

    # Iterate through the images and create a list of tuples (image, annotations)
    for image_count, image_info in enumerate(coco_annotations['images']):
        image_id = image_info['id']
        # Load and preprocess the image (you can use your own image loading logic)
        image = load_and_preprocess_image(os.path.join(dataset_folder, image_info['file_name']), preprocess)
        annotations = annotations_by_image.get(image_id, [])
        if len(annotations) > 0:
            annotations[0]['orig_img_dims'] = (image_info['height'], image_info['width'])

            # Add the image and annotations to the current batch
            batch_images.append(image)
            batch_annotations.append(annotations)

            # Check if the current batch is of the desired batch size
            if len(batch_images) == batch_size:
                # Yield the current batch
                yield np.array(batch_images), batch_annotations

                # Reset the batch lists for the next batch
                batch_images = []
                batch_annotations = []

        # After processing all images, yield any remaining images in the last batch
        if len(batch_images) > 0 and (total_images == image_count + 1):
            yield np.array(batch_images), batch_annotations