import numpy as np
import torch
import torch_npu
import cv2
import os
from pathlib import Path
import glob
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def filter_redundant_masks(masks, iou_thresh=0.9, contain_thresh=0.9):
    merged = []
    used = [False] * len(masks)
    for i, mask_i in enumerate(masks):
        if used[i]:
            continue
        merged_mask = mask_i.copy()
        used[i] = True
        for j in range(i + 1, len(masks)):
            if used[j]:
                continue
            mask_j = masks[j]
            inter = np.logical_and(merged_mask > 0, mask_j > 0).sum()
            union = np.logical_or(merged_mask > 0, mask_j > 0).sum()
            iou = inter / union if union > 0 else 0
            area_i = (merged_mask > 0).sum()
            area_j = (mask_j > 0).sum()
            contain_ratio = inter / min(area_i, area_j)
            if iou > iou_thresh or contain_ratio > contain_thresh:
                merged_mask = np.logical_or(merged_mask > 0, mask_j > 0).astype(np.uint8) * 255
                used[j] = True
        merged.append(merged_mask)
    return merged


def postprocess_masks(masks, min_area_ratio=0.002, smooth=True):
    processed_masks = []
    if len(masks) == 0:
        return processed_masks
    h, w = masks[0]['segmentation'].shape
    min_area = h * w * min_area_ratio
    for ann in masks:
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


def erode_mask(white_img, H,W, kernel_size_ratio, erode_iter=2, dilate_iter=2):
    short_side = min(H, W)
    if kernel_size_ratio > 0:
        ksize = max(1, int(short_side * kernel_size_ratio))
        if ksize % 2 == 0:
            ksize += 1
    else:
        ksize = 7
    bg = (white_img[..., 0] == 255) & (white_img[..., 1] == 255) & (white_img[..., 2] == 255)
    bin_img = np.where(bg, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    eroded = cv2.erode(bin_img, kernel, iterations=erode_iter)
    morphed = cv2.dilate(eroded, kernel, iterations=dilate_iter)
    discarded_mask = np.logical_and(bin_img == 255, morphed == 0).astype(np.uint8) * 255
    return morphed, discarded_mask

def _safe_imwrite(path, img):
    ext = os.path.splitext(path)[1]
    if ext == "":
        ext = ".png"
        path = path + ext
    _, buf = cv2.imencode(ext, img)
    buf.tofile(path)


def process_others_stage(dict, remain_RGB, base):
    sam_checkpoint = dict['sam_remain_sam_checkpoint']
    model_type = dict['sam_remain_model_type']
    device = dict['device']

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=dict['sam_remain_points_per_side'],
        pred_iou_thresh=dict['sam_remain_pred_iou_thresh'],
        stability_score_thresh=dict['sam_remain_stability_score_thresh'],
        crop_n_layers=dict['sam_remain_crop_n_layers'],
        crop_nms_thresh=dict['sam_remain_crop_nms_thresh'],
        crop_overlap_ratio=dict['sam_remain_crop_overlap_ratio'],
        min_mask_region_area=dict['sam_remain_min_mask_region_area'],
        output_mode=dict['sam_remain_output_mode']
    )
    height, width = remain_RGB.shape[:2]
    mask_others = []
    output_dir = dict['output_dir']
    max_side = max(height, width)
    size_rate = 1
    resize_max_side = dict['sam_remain_resize_max_side']
    if max_side > resize_max_side:
        size_rate = resize_max_side / max_side

    # initialization
    processed_mask_total = np.zeros((height, width), dtype=np.uint8)
    background_mask = cv2.inRange(remain_RGB, (255, 255, 255), (255, 255, 255))

    # update processed_mask_total
    processed_mask_total = cv2.bitwise_or(processed_mask_total, background_mask)
    current_remain = remain_RGB.copy()
    loop = 1

    # multiple splits
    while loop < 5:
        print(f"\n[loop {loop}] Start multiple segmentation...")

        resized_img = cv2.resize(current_remain, (int(width * size_rate), int(height * size_rate)))
        masks = mask_generator_2.generate(resized_img)
        masks = postprocess_masks(masks)
        if len(masks) <= 1:
            print(f" [loop {loop}] Number of split instances={len(masks)}，end")
            break

        filtered_masks = []
        for mask in masks:
            mask_fullsize = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            inter = cv2.bitwise_and(mask_fullsize, processed_mask_total)
            union = cv2.bitwise_or(mask_fullsize, processed_mask_total)
            iou = cv2.countNonZero(inter) / (cv2.countNonZero(union) + 1e-6)
            contain = cv2.countNonZero(inter) / (cv2.countNonZero(mask_fullsize) + 1e-6)
            if iou < 0.55 and contain < 0.55:
                filtered_masks.append(mask)

        masks = sorted(filtered_masks, key=lambda x: cv2.countNonZero(x), reverse=True)
        masks = filter_redundant_masks(masks, iou_thresh=dict['sam_remain_iou_filter_thresh'], contain_thresh=dict['sam_remain_iou_contain_thresh'])

        valid_masks = []
        for mask in masks:
            mask_fullsize = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            white_area = cv2.countNonZero(cv2.bitwise_and(mask_fullsize, background_mask))
            total_area = cv2.countNonZero(mask_fullsize)
            white_ratio = white_area / (total_area + 1e-6)

            if white_ratio < 0.5:
                valid_masks.append(mask)
            else:
                print(f"Abandoning a mask, the proportion of white areas {white_ratio:. 2f} exceeds the threshold of 0.5")

        masks = valid_masks

        mask_cut_output_dir = Path(output_dir) / "remain_instance_bin"
        os.makedirs(mask_cut_output_dir, exist_ok=True)

        for i, mask in enumerate(masks):
            mask_fullsize = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            save_name = f"{base}_loop{loop}_mask{i + 1}.png"
            mask_others.append(mask_fullsize)
            cv2.imencode('.png', mask_fullsize)[1].tofile(
                os.path.join(output_dir, "remain_instance_bin", save_name)
            )

        # 涂白
        segmented_rgb = current_remain.copy()
        for mask in masks:
            mask_fullsize = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            segmented_rgb[mask_fullsize == 255] = (255, 255, 255)
        

        # 腐蚀 + 更新
        morphed_mask, discarded_mask = erode_mask(segmented_rgb, height, width, dict['sam_remain_erode_kernel_size_ratio'], erode_iter=dict['sam_remain_erode_erode_iter'], dilate_iter=dict['sam_remain_erode_dilate_iter'])
        processed_mask_total = cv2.bitwise_or(processed_mask_total, discarded_mask)

        morphed_rgb = segmented_rgb.copy()
        morphed_rgb[morphed_mask == 0] = (255, 255, 255)
        current_remain = morphed_rgb.copy()

        print(f" [loop {loop}] sam instance counts={len(masks)}")
        loop += 1

    # Small area filtration
    print("\nwhile Cycle ends, start small area filtering...")
    gray = cv2.cvtColor(current_remain, cv2.COLOR_RGB2GRAY)
    _, bin_mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_discarded_mask = np.zeros_like(bin_mask)
    remain_rgb = current_remain.copy()
    min_area = int(dict['sam_remain_min_area_ratio'] * width * height)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            cv2.drawContours(final_discarded_mask, [cnt], -1, 255, -1)
            cv2.drawContours(remain_rgb, [cnt], -1, (255, 255, 255), -1)

    gray = cv2.cvtColor(remain_rgb, cv2.COLOR_RGB2GRAY)
    _, bin_mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    mask_others.append(bin_mask)
    union_mask_first = np.zeros((height, width), dtype=np.uint8)
    output_dir = dict['output_dir']
    first_instance_dir = Path(output_dir) / "remain_instance_bin"
    first_instance_dir.mkdir(parents=True, exist_ok=True)
    for mask in mask_others:
        union_mask_first = cv2.bitwise_or(union_mask_first, mask)
    union_mask_first_path = os.path.join(first_instance_dir, f"{base}_remain_union_mask.png")
    _safe_imwrite(union_mask_first_path, union_mask_first)

    return mask_others

