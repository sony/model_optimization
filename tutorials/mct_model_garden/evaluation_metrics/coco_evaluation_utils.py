# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
from enum import Enum
import numpy as np
from typing import List

class BoxFormat(Enum):
    YMIM_XMIN_YMAX_XMAX = 'ymin_xmin_ymax_xmax'
    XMIM_YMIN_XMAX_YMAX = 'xmin_ymin_xmax_ymax'
    XMIN_YMIN_W_H = 'xmin_ymin_width_height'
    XC_YC_W_H = 'xc_yc_width_height'


def convert_to_ymin_xmin_ymax_xmax_format(boxes, orig_format: BoxFormat):
    """
    changes the box from one format to another (XMIN_YMIN_W_H --> YMIM_XMIN_YMAX_XMAX )
    also support in same format mode (returns the same format)

    :param boxes:
    :param orig_format:
    :return: box in format YMIM_XMIN_YMAX_XMAX
    """
    if len(boxes) == 0:
        return boxes
    elif orig_format == BoxFormat.YMIM_XMIN_YMAX_XMAX:
        return boxes
    elif orig_format == BoxFormat.XMIN_YMIN_W_H:
        boxes[:, 2] += boxes[:, 0]  # convert width to xmax
        boxes[:, 3] += boxes[:, 1]  # convert height to ymax
        boxes[:, 0], boxes[:, 1] = boxes[:, 1], boxes[:, 0].copy()  # swap xmin, ymin columns
        boxes[:, 2], boxes[:, 3] = boxes[:, 3], boxes[:, 2].copy()  # swap xmax, ymax columns
        return boxes
    elif orig_format == BoxFormat.XMIM_YMIN_XMAX_YMAX:
        boxes[:, 0], boxes[:, 1] = boxes[:, 1], boxes[:, 0].copy()  # swap xmin, ymin columns
        boxes[:, 2], boxes[:, 3] = boxes[:, 3], boxes[:, 2].copy()  # swap xmax, ymax columns
        return boxes
    elif orig_format == BoxFormat.XC_YC_W_H:
        new_boxes = np.copy(boxes)
        new_boxes[:, 0] = boxes[:, 1] - boxes[:, 3] / 2  # top left y
        new_boxes[:, 1] = boxes[:, 0] - boxes[:, 2] / 2  # top left x
        new_boxes[:, 2] = boxes[:, 1] + boxes[:, 3] / 2  # bottom right y
        new_boxes[:, 3] = boxes[:, 0] + boxes[:, 2] / 2  # bottom right x
        return new_boxes
    else:
        raise Exception("Unsupported boxes format")

def clip_boxes(boxes: np.ndarray, h: int, w: int) -> np.ndarray:
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


def scale_boxes(boxes: np.ndarray, h_image: int, w_image: int, h_model: int, w_model: int, preserve_aspect_ratio: bool,
                align_center: bool = True, normalized: bool = True) -> np.ndarray:
    """
    Scale and offset bounding boxes based on model output size and original image size.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes in format [y_min, x_min, y_max, x_max].
        h_image (int): Original image height.
        w_image (int): Original image width.
        h_model (int): Model output height.
        w_model (int): Model output width.
        preserve_aspect_ratio (bool): Whether to preserve image aspect ratio during scaling
        align_center (bool): Whether to center the bounding boxes after scaling
        normalized (bool): Whether treats bounding box coordinates as normalized (i.e., in the range [0, 1])

    Returns:
        numpy.ndarray: Scaled and offset bounding boxes.
    """
    deltaH, deltaW = 0, 0
    H, W = h_model, w_model
    scale_H, scale_W = h_image / H, w_image / W

    if preserve_aspect_ratio:
        scale_H = scale_W = max(h_image / H, w_image / W)
        H_tag = int(np.round(h_image / scale_H))
        W_tag = int(np.round(w_image / scale_W))
        if align_center:
            deltaH, deltaW = int((H - H_tag) / 2), int((W - W_tag) / 2)

    nh, nw = (H, W) if normalized else (1, 1)

    # Scale and offset boxes
    boxes[..., 0] = (boxes[..., 0] * nh - deltaH) * scale_H
    boxes[..., 1] = (boxes[..., 1] * nw - deltaW) * scale_W
    boxes[..., 2] = (boxes[..., 2] * nh - deltaH) * scale_H
    boxes[..., 3] = (boxes[..., 3] * nw - deltaW) * scale_W

    # Clip boxes
    boxes = clip_boxes(boxes, h_image, w_image)

    return boxes


def scale_coords(kpts: np.ndarray, h_image: int, w_image: int, h_model: int, w_model: int, preserve_aspect_ratio: bool) -> np.ndarray:
    """
    Scale and offset keypoints based on model output size and original image size.

    Args:
        kpts (numpy.ndarray): Array of bounding keypoints in format [..., 17, 3]  where the last dim is (x, y, visible).
        h_image (int): Original image height.
        w_image (int): Original image width.
        h_model (int): Model output height.
        w_model (int): Model output width.
        preserve_aspect_ratio (bool): Whether to preserve image aspect ratio during scaling

    Returns:
        numpy.ndarray: Scaled and offset bounding boxes.
    """
    deltaH, deltaW = 0, 0
    H, W = h_model, w_model
    scale_H, scale_W = h_image / H, w_image / W

    if preserve_aspect_ratio:
        scale_H = scale_W = max(h_image / H, w_image / W)
        H_tag = int(np.round(h_image / scale_H))
        W_tag = int(np.round(w_image / scale_W))
        deltaH, deltaW = int((H - H_tag) / 2), int((W - W_tag) / 2)

    # Scale and offset boxes
    kpts[..., 0] = (kpts[..., 0]  - deltaH) * scale_H
    kpts[..., 1] = (kpts[..., 1] - deltaW) * scale_W

    # Clip boxes
    kpts = clip_coords(kpts, h_image, w_image)

    return kpts

def clip_coords(kpts: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Clip keypoints to stay within the image boundaries.

    Args:
        kpts (numpy.ndarray): Array of bounding keypoints in format [..., 17, 3]  where the last dim is (x, y, visible).
        h (int): Height of the image.
        w (int): Width of the image.

    Returns:
        numpy.ndarray: Clipped bounding boxes.
    """
    kpts[..., 0] = np.clip(kpts[..., 0], a_min=0, a_max=h)
    kpts[..., 1] = np.clip(kpts[..., 1], a_min=0, a_max=w)
    return kpts


def nms(dets: np.ndarray, scores: np.ndarray, iou_thres: float = 0.5, max_out_dets: int = 300) -> List[int]:
    """
    Perform Non-Maximum Suppression (NMS) on detected bounding boxes.

    Args:
        dets (np.ndarray): Array of bounding box coordinates of shape (N, 4) representing [y1, x1, y2, x2].
        scores (np.ndarray): Array of confidence scores associated with each bounding box.
        iou_thres (float, optional): IoU threshold for NMS. Default is 0.5.
        max_out_dets (int, optional): Maximum number of output detections to keep. Default is 300.

    Returns:
        List[int]: List of indices representing the indices of the bounding boxes to keep after NMS.

    """
    y1, x1 = dets[:, 0], dets[:, 1]
    y2, x2 = dets[:, 2], dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]

    return keep[:max_out_dets]

def combined_nms(batch_boxes, batch_scores, iou_thres: float = 0.5, conf: float = 0.001, max_out_dets: int = 300):

    """
    Performs combined Non-Maximum Suppression (NMS) on batches of bounding boxes and scores.

    Parameters:
    batch_boxes (List[np.ndarray]): A list of arrays, where each array contains bounding boxes for a batch.
    batch_scores (List[np.ndarray]): A list of arrays, where each array contains scores for the corresponding bounding boxes.
    iou_thres (float): Intersection over Union (IoU) threshold for NMS. Defaults to 0.5.
    conf (float): Confidence threshold for filtering boxes. Defaults to 0.001.
    max_out_dets (int): Maximum number of output detections per image. Defaults to 300.

    Returns:
    List[Tuple[np.ndarray, np.ndarray, np.ndarray]]: A list of tuples for each batch, where each tuple contains:
        - nms_bbox: Array of bounding boxes after NMS.
        - nms_scores: Array of scores after NMS.
        - nms_classes: Array of class IDs after NMS.
    """
    nms_results = []
    for boxes, scores in zip(batch_boxes, batch_scores):

        xc = np.argmax(scores, 1)
        xs = np.amax(scores, 1)
        x = np.concatenate([boxes, np.expand_dims(xs, 1), np.expand_dims(xc, 1)], 1)

        xi = xs > conf
        x = x[xi]

        x = x[np.argsort(-x[:, 4])[:8400]]
        scores = x[:, 4]
        x[..., :4] = convert_to_ymin_xmin_ymax_xmax_format(x[..., :4], BoxFormat.XC_YC_W_H)
        offset = x[:, 5] * 640
        boxes = x[..., :4] + np.expand_dims(offset, 1)

        # Original post-processing part
        valid_indexs = nms(boxes, scores, iou_thres=iou_thres, max_out_dets=max_out_dets)
        x = x[valid_indexs]
        nms_classes = x[:, 5]
        nms_bbox = x[:, :4]
        nms_scores = x[:, 4]

        nms_results.append((nms_bbox, nms_scores, nms_classes))

    return nms_results

