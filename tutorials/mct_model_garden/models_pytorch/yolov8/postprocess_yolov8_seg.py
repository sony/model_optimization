# ------------------------------------------------------------------------------
# This file contains code from the Ultralytics repository (YOLOv8)
# Copyright (C) 2024  Ultralytics
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------

"""
Part of this code was based on Ultralytics implementation. For more details, refer to the original repository:
https://github.com/ultralytics/ultralytics
"""
from typing import List
import numpy as np
import cv2
from typing import Tuple

from tutorials.mct_model_garden.models_pytorch.yolov8.yolov8_postprocess import nms


def combined_nms_seg(batch_boxes, batch_scores, batch_masks, iou_thres: float = 0.3, conf: float = 0.1, max_out_dets: int = 300):
    """
    Perform combined Non-Maximum Suppression (NMS) and segmentation mask processing for batched inputs.

    This function processes batches of bounding boxes, confidence scores, and segmentation masks by applying
    class-wise NMS to filter out overlapping boxes based on their Intersection over Union (IoU) and confidence scores.
    It also filters detections based on a confidence threshold and returns the final bounding boxes, scores, class indices,
    and corresponding segmentation masks.

    Args:
        batch_boxes (List[np.ndarray]): List of arrays, each containing bounding boxes for an image in the batch.
                                        Each array is of shape (N, 4), where N is the number of detections,
                                        and each box is represented as [y1, x1, y2, x2].
        batch_scores (List[np.ndarray]): List of arrays, each containing confidence scores for detections in an image.
                                         Each array is of shape (N, num_classes), where N is the number of detections.
        batch_masks (List[np.ndarray]): List of arrays, each containing segmentation masks for detections in an image.
                                        Each array is of shape (num_classes, H, W), where H and W are the dimensions
                                        of the output mask.
        iou_thres (float, optional): IoU threshold for NMS. Default is 0.3.
        conf (float, optional): Confidence threshold to filter detections. Default is 0.1.
        max_out_dets (int, optional): Maximum number of output detections to keep after NMS. Default is 300.

    Returns:
        List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]: A list of tuples, each containing:
            - Bounding boxes of the final detections (shape: (K, 4))
            - Confidence scores of the final detections (shape: (K,))
            - Class indices of the final detections (shape: (K,))
            - Segmentation masks corresponding to the final detections (shape: (K, H, W))
            where K is the number of final detections kept after NMS and confidence filtering.
    """
    nms_results = []
    for boxes, scores, masks in zip(batch_boxes, batch_scores, batch_masks):
        # Compute maximum scores and corresponding class indices
        class_indices = np.argmax(scores, axis=1)
        max_scores = np.amax(scores, axis=1)
        detections = np.concatenate([boxes, np.expand_dims(max_scores, axis=1), np.expand_dims(class_indices, axis=1)], axis=1)

        masks = np.transpose(masks, (1, 0))
        valid_detections = max_scores > conf
        detections = detections[valid_detections]
        masks = masks[valid_detections]

        if len(detections) == 0:
            nms_results.append((np.array([]), np.array([]), np.array([]), np.array([[]])))
            continue

        # Sort detections by score in descending order
        sorted_indices = np.argsort(-detections[:, 4])
        detections = detections[sorted_indices]
        masks = masks[sorted_indices]

        # Perform class-wise NMS
        unique_classes = np.unique(detections[:, 5])
        all_indices = []

        for cls in unique_classes:
            cls_indices = np.where(detections[:, 5] == cls)[0]
            cls_boxes = detections[cls_indices, :4]
            cls_scores = detections[cls_indices, 4]
            cls_valid_indices = nms(cls_boxes, cls_scores, iou_thres=iou_thres, max_out_dets=len(cls_indices))  # Use all available for NMS
            all_indices.extend(cls_indices[cls_valid_indices])

        if len(all_indices) == 0:
            nms_results.append((np.array([]), np.array([]), np.array([]), np.array([[]])))
            continue

        # Sort all indices by score and limit to max_out_dets
        all_indices = np.array(all_indices)
        all_indices = all_indices[np.argsort(-detections[all_indices, 4])]
        final_indices = all_indices[:max_out_dets]

        final_detections = detections[final_indices]
        final_masks = masks[final_indices]

        # Extract class indices, bounding boxes, and scores
        nms_classes = final_detections[:, 5]
        nms_bbox = final_detections[:, :4]
        nms_scores = final_detections[:, 4]

        # Append results including masks
        nms_results.append((nms_bbox, nms_scores, nms_classes, final_masks))

    return nms_results

 
def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box

    Args:
      masks (numpy.ndarray): [h, w, n] tensor of masks
      boxes (numpy.ndarray): [n, 4] tensor of bbox coordinates in relative point form

    Returns:
      (numpy.ndarray): The masks are being cropped to the bounding box.
    """
    n, w, h = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
    c = np.arange(h, dtype=np.float32)[None, None, :]
    r = np.arange(w, dtype=np.float32)[None, :, None] 

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def calculate_padding(original_shape, target_shape):
    """
    Calculate the padding needed to center the image in the target shape and the scale factor used for resizing.

    Args:
      original_shape (tuple): The height and width of the original image.
      target_shape (tuple): The desired height and width for scaling the image.

    Returns:
      tuple: A tuple containing the padding widths (pad_width, pad_height) and the scale factor.
    """
    orig_height, orig_width = original_shape[:2]
    target_height, target_width = target_shape
    larger_dim = max(orig_height, orig_width)
    if not target_height==target_width:
        print('model input must be square')
    scale = target_height/larger_dim

    scaled_width = int(orig_width * scale)
    scaled_height = int(orig_height * scale)

    pad_width = max((target_width - scaled_width) // 2, 0)
    pad_height = max((target_height - scaled_height) // 2, 0)
    
    return pad_width, pad_height, scale



def crop_to_original(mask, pad_width, pad_height, original_shape, scale):
    """
    Crop the mask to the original image dimensions after padding and scaling adjustments.

    Args:
      mask (numpy.ndarray): The mask to be cropped.
      pad_width (int): The padding width applied to the mask.
      pad_height (int): The padding height applied to the mask.
      original_shape (tuple): The original dimensions of the image (height, width).
      scale (float): The scaling factor applied to the original dimensions.

    Returns:
      numpy.ndarray: The cropped mask.
    """
    end_height = min(pad_height + (original_shape[0]*scale), mask.shape[0])
    end_width = min(pad_width + (original_shape[1]*scale), mask.shape[1])
    cropped_mask = mask[int(pad_height):int(end_height), int(pad_width):int(end_width)]
    return cropped_mask

def process_masks(masks, boxes, orig_img_shape, model_input_size):
    """
    Adjusts and crops masks for detected objects to fit original image dimensions. 

    Args:
      masks (numpy.ndarray): Input masks to be processed.
      boxes (numpy.ndarray): Bounding boxes for cropping masks.
      orig_img_shape (tuple): Original dimensions of the image.
      model_input_size (tuple): Input size required by the model.

    Returns:
      numpy.ndarray: Processed masks adjusted and cropped to fit the original image dimensions.

    Processing Steps:
    1. Calculate padding and scaling for model input size adjustment.
    2. Apply sigmoid to normalize mask values.
    3. Resize masks to model input size.
    4. Crop masks to original dimensions using calculated padding.
    5. Resize cropped masks to original dimensions.
    6. Crop masks per bounding boxes for individual objects.
    """
    if masks.size == 0:  # Check if the masks array is empty
        return np.array([]) 
    pad_width, pad_height, scale = calculate_padding(orig_img_shape, model_input_size)
    masks = 1 / (1 + np.exp(-masks))
    orig_height, orig_width = orig_img_shape[:2]
    masks = np.transpose(masks, (2, 1, 0))  # Change to HWC format
    masks = cv2.resize(masks, model_input_size, interpolation=cv2.INTER_LINEAR)
      
    masks = np.expand_dims(masks, -1) if len(masks.shape) == 2 else masks
    masks = np.transpose(masks, (2, 1, 0))  # Change back to CHW format
    #crop masks based on padding
    masks = [crop_to_original(mask, pad_width, pad_height, orig_img_shape, scale) for mask in masks]
    masks = np.stack(masks, axis=0)
    
    masks = np.transpose(masks, (2, 1, 0))  # Change to HWC format
    masks = cv2.resize(masks,  (orig_height, orig_width), interpolation=cv2.INTER_LINEAR)
    masks = np.expand_dims(masks, -1) if len(masks.shape) == 2 else masks
    masks = np.transpose(masks, (2, 1, 0))  # Change back to CHW format
    # Crop masks based on bounding boxes
    masks = crop_mask(masks, boxes)

    return masks


def postprocess_yolov8_inst_seg(outputs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                                conf: float = 0.1,
                                iou_thres: float = 0.3,
                                max_out_dets: int = 300) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Post-processes the outputs of a YOLOv8 instance segmentation model.

    Args:
        outputs (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): Tuple containing the outputs from the model:
            - y_bb: Bounding box coordinates
            - y_cls: Class probabilities
            - ymask_weights: Weights for combining masks
            - y_masks: Segmentation masks
        conf (float): Confidence threshold for filtering detections.
        iou_thres (float): IOU threshold for non-maximum suppression.
        max_out_dets (int): Maximum number of detections to return.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing:
            - nms_bbox: Bounding boxes after NMS.
            - nms_scores: Scores of the bounding boxes.
            - nms_classes: Class IDs of the bounding boxes.
            - final_masks: Combined segmentation masks after applying mask weights.
    """
    
    
    y_bb, y_cls, ymask_weights, y_masks = outputs
    y_bb= np.transpose(y_bb, (0,2,1))
    y_cls= np.transpose(y_cls, (0,2,1))
    y_bb = y_bb * 640 #image size
    detect_out = np.concatenate((y_bb, y_cls), 1)
    xd = detect_out.transpose([0, 2, 1])
    nms_bbox, nms_scores, nms_classes, ymask_weights = combined_nms_seg(xd[..., :4], xd[..., 4:84], ymask_weights, iou_thres, conf, max_out_dets)[0]
    y_masks = y_masks.squeeze(0)

    if ymask_weights.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    ymask_weights = ymask_weights.transpose(1, 0)

    final_masks = np.tensordot(ymask_weights, y_masks, axes=([0], [0]))

    return nms_bbox, nms_scores, nms_classes, final_masks


