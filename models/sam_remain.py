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

def erode_mask_bin(bin_mask, kernel_size_ratio=0.01, erode_iter=2, dilate_iter=2):
    # Perform corrosion and expansion operations on the mask, and return the corroded area
    h, w = bin_mask.shape
    short_side = min(h, w)
    ksize = max(3, int(short_side * kernel_size_ratio))
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

    eroded = cv2.erode(bin_mask, kernel, iterations=erode_iter)
    morphed = cv2.dilate(eroded, kernel, iterations=dilate_iter)

    discarded = np.logical_and(bin_mask == 255, morphed == 0).astype(np.uint8) * 255
    return morphed, discarded

def _safe_imwrite(path, img):
    ext = os.path.splitext(path)[1]
    if ext == "":
        ext = ".png"
        path = path + ext
    _, buf = cv2.imencode(ext, img)
    buf.tofile(path)

def process_others_stage(dict, first_remain, image, base):
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

    height, width = first_remain.shape[:2]
    output_dir = Path(dict['output_dir'])
    resize_max_side = dict['sam_remain_resize_max_side']

    # Initialize processed_mask_total (black areas are recognized)
    gray_first = cv2.cvtColor(first_remain, cv2.COLOR_RGB2GRAY) if len(first_remain.shape) == 3 else first_remain
    _, processed_mask_total = cv2.threshold(gray_first, 10, 255, cv2.THRESH_BINARY_INV)

    # remain area
    remain_mask = cv2.bitwise_not(processed_mask_total)
    save_path = f"{base}_save_mask.png"
    _safe_imwrite(str(save_path), remain_mask)

    # SAM segmentation image (white areas have been highlighted)
    remain_rgb = image.copy()
    remain_rgb[first_remain == 0] = (255, 255, 255)
    remain_rgb_dir = output_dir / "remain_rgb"
    remain_rgb_dir.mkdir(parents=True, exist_ok=True)
    save_path = remain_rgb_dir / f"{base}_save_mask.png"
    _safe_imwrite(str(save_path), remain_mask)

    # save remain_rgb
    initial_remain_rgb_path = remain_rgb_dir / f"{base}_remain_rgb.png"
    _safe_imwrite(str(initial_remain_rgb_path), cv2.cvtColor(remain_rgb, cv2.COLOR_RGB2BGR))
    print(f"Initial saved remain_rgb -> {initial_remain_rgb_path}")

    mask_others = []
    loop = 1
    while loop <= 5:
        print(f"\n[loop {loop}] SAM seg...")

        # Image scaling
        max_side = max(height, width)
        size_rate = min(1.0, resize_max_side / max_side)
        resized_img = cv2.resize(remain_rgb, (int(width * size_rate), int(height * size_rate)))

        # SAM seg
        masks = mask_generator_2.generate(resized_img)
        masks = postprocess_masks(masks)
        if len(masks) <= 1:
            print(f"️ [loop {loop}] instance count={len(masks)}，end")
            break

        # Enlarge back to the original image size
        masks = [cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST) for m in masks]

        # Remove masks that overlap heavily with processed_mask_total
        filtered = []
        for m in masks:
            inter = cv2.bitwise_and(m, processed_mask_total)
            union = cv2.bitwise_or(m, processed_mask_total)
            iou = cv2.countNonZero(inter) / (cv2.countNonZero(union) + 1e-6)
            contain = cv2.countNonZero(inter) / (cv2.countNonZero(m) + 1e-6)
            if iou < 0.7 and contain < 0.7:
                filtered.append(m)

        refined_masks = []
        min_area_loop = int(dict['sam_remain_min_area_ratio_loop'] * width * height)

        for mask in filtered:
            mask = (mask > 127).astype(np.uint8) * 255
            processed_mask_total = (processed_mask_total > 127).astype(np.uint8) * 255

            # Block overlapping areas
            mask_no_overlap = cv2.bitwise_and(mask, cv2.bitwise_not(processed_mask_total))

            # Find the outline of the white area
            contours, _ = cv2.findContours(mask_no_overlap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Small area painted black
                if area < min_area_loop:
                    cv2.drawContours(mask_no_overlap, [cnt], -1, 0, thickness=cv2.FILLED)

            # Determine if there are still valid white areas
            if np.sum(mask_no_overlap > 0) == 0:
                continue

            # update processed_mask_total
            processed_mask_total = cv2.bitwise_or(processed_mask_total, mask_no_overlap)
            refined_masks.append(mask_no_overlap)

        masks = filter_redundant_masks(refined_masks,
                                       iou_thresh=dict['sam_remain_iou_filter_thresh'],
                                       contain_thresh=dict['sam_remain_iou_contain_thresh'])

        if len(masks) <= 0:
            print(f" [loop {loop}] instance count={len(masks)}，end")
            break

        # update processed_mask_total
        for m in masks:
            #processed_mask_total = cv2.bitwise_or(processed_mask_total, m)
            mask_others.append(m)


        # Use mask corrosion
        morphed_mask, discarded_mask = erode_mask_bin(
            cv2.bitwise_not(processed_mask_total),
            kernel_size_ratio=dict['sam_remain_erode_kernel_size_ratio'],
            erode_iter=dict['sam_remain_erode_erode_iter'],
            dilate_iter=dict['sam_remain_erode_dilate_iter']
        )

        processed_mask_total = cv2.bitwise_or(processed_mask_total, discarded_mask)

        remain_rgb = image.copy()
        remain_rgb[processed_mask_total == 255] = (255, 255, 255)

        print(f" [loop {loop}] new instance count={len(masks)}")
        loop += 1

    # Small area filtration
    print("\n[post-processing] Small area filtration...")
    processed_mask_total = (processed_mask_total.astype(np.uint8) > 0).astype(np.uint8) * 255

    # Reverse to obtain unrecognized areas
    final_remaining = cv2.bitwise_not(processed_mask_total)
    final_remaining, _ = erode_mask_bin(
        final_remaining,
        kernel_size_ratio=dict['sam_remain_erode_kernel_size_ratio'],
        erode_iter=dict['sam_remain_erode_erode_iter'],
        dilate_iter=dict['sam_remain_erode_dilate_iter']
    )

    # Small area filtering parameters (in pixels)
    min_area = int(dict['sam_remain_min_area_ratio'] * width * height)

    # Find connected domains/contours and remove small areas
    contours, _ = cv2.findContours(final_remaining.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_remaining_clean = np.zeros_like(final_remaining)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            # Retain larger connectivity areas
            cv2.drawContours(final_remaining_clean, [cnt], -1, 255, thickness=cv2.FILLED)

    # If there are still white areas after filtering, add them to mask_others
    if np.count_nonzero(final_remaining_clean) > 0:
        mask_others.append(final_remaining_clean)

    # Calculate and save the union according to the original logic
    union_mask_first = np.zeros((height, width), dtype=np.uint8)
    first_instance_dir = Path(output_dir) / "remain_instance_bin"
    first_instance_dir.mkdir(parents=True, exist_ok=True)
    for mask in mask_others:
        union_mask_first = cv2.bitwise_or(union_mask_first, mask)
    union_mask_first_path = first_instance_dir / f"{base}_remain_union_mask.png"
    print(union_mask_first_path)
    _safe_imwrite(str(union_mask_first_path), union_mask_first)
    loop_dir = output_dir / "remain_instance_bin"
    loop_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(mask_others):
        save_path = loop_dir / f"{base}_mask{i + 1}.png"
        _safe_imwrite(str(save_path), m)
    print(f"All instances have been saved and merged -> {union_mask_first_path}")
    return mask_others

