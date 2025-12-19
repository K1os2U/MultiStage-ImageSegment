import os
import glob
from pathlib import Path
import numpy as np
import cv2
import torch
import torch_npu
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def postprocess_masks(anns, min_area_ratio=0.002, smooth=True):
    processed_masks = []
    if len(anns) == 0:
        return processed_masks
    h, w = anns[0]['segmentation'].shape
    min_area = h * w * min_area_ratio
    for ann in anns:
        mask = ann['segmentation'].astype(np.uint8) * 255
        if cv2.countNonZero(mask) < min_area:
            continue
        if smooth:
            kernel_size = max(5, int(min(h, w) * 0.005) | 1)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        processed_masks.append(mask)
    return processed_masks

def _safe_imwrite(path, img):
    ext = os.path.splitext(path)[1]
    if ext == "":
        ext = ".png"
        path = path + ext
    _, buf = cv2.imencode(ext, img)
    buf.tofile(path)


def process_first_stage(dict, img_path):
    model_type = dict['sam_first_model_type']
    sam_checkpoint = dict['sam_first_sam_checkpoint']
    device = dict['device']
    # 加载 SAM 
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=dict['sam_first_points_per_side'],
        pred_iou_thresh=dict['sam_first_pred_iou_thresh'],
        stability_score_thresh=dict['sam_first_pred_iou_thresh'],
        crop_n_layers=dict['sam_first_crop_n_layers'],
        crop_nms_thresh=dict['sam_first_crop_nms_thresh'],
        crop_overlap_ratio=dict['sam_first_crop_overlap_ratio'],
        min_mask_region_area=dict['sam_first_min_mask_region_area'],
        output_mode=dict['sam_first_output_mode'],
    )

    # load image
    image_bgr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    max_side = max(height, width)
    size_rate = 1
    resize_max_side = dict['sam_first_resize_max_side']
    if max_side > resize_max_side:
        size_rate = resize_max_side / max_side

    # Initial segmentation
    base = os.path.splitext(os.path.basename(img_path))[0]
    resized_image = cv2.resize(image, (max(1, int(width * size_rate)), max(1, int(height * size_rate))))
    anns = mask_generator.generate(resized_image)
    masks_1st = postprocess_masks(anns,min_area_ratio=0, smooth=False)
    # Sort by Area
    masks_1st = [
        cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        for mask in masks_1st
    ]

    return masks_1st, image, base

