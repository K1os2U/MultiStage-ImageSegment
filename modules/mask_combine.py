import os
import glob
from pathlib import Path
import numpy as np
import cv2
import torch
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
            contain_ratio = inter / min(area_i, area_j) if min(area_i, area_j) > 0 else 0

            if iou > iou_thresh or contain_ratio > contain_thresh:
                merged_mask = np.logical_or(merged_mask > 0, mask_j > 0).astype(np.uint8) * 255
                used[j] = True

        merged.append(merged_mask)

    return merged

def erode_mask(white_img, kernel_size=7, erode_iter=2, dilate_iter=2):
    
    bg = (white_img[..., 0] == 255) & (white_img[..., 1] == 255) & (white_img[..., 2] == 255)
    bin_img = np.where(bg, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
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

# Main function: Split and save once first_instance_bin & first_remain_bin & first_remain_RGB
def process_combine( dict, masks_1st,image, base):
    min_area_ratio = dict['mask_combine_min_area_ratio']
    output_dir = dict['output_dir']
    height, width = image.shape[:2]
    min_area = int(height * width * min_area_ratio)
    processed_masks = []
    for mask in masks_1st:
        if cv2.countNonZero(mask) < min_area:
            continue
        kernel_size = max(5, int(min(height, width) * 0.005) | 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        processed_masks.append(mask)

    masks_1st = processed_masks
    masks_1st = sorted(masks_1st, key=lambda x: cv2.countNonZero(x), reverse=True)
    masks_1st = filter_redundant_masks(masks_1st, iou_thresh=dict['mask_combine_iou_filter_thresh'], contain_thresh=dict['mask_combine_contain_thresh'])

    covered_mask = np.zeros((height, width), dtype=np.uint8)

    # Save each instance (binary, placed in directory: first_instance bin)
    mask_first = []
    first_instance_dir = Path(output_dir) / "first_instance_bin"
    first_instance_dir.mkdir(parents=True, exist_ok=True)
    for i, mask in enumerate(masks_1st):
        save_name = f"{base}_first_mask{i + 1}.png"
        _safe_imwrite(os.path.join(first_instance_dir, save_name), mask)
        covered_mask[mask == 255] = 255
        mask_first.append(mask)
        print(f" Save the initial split instance {i + 1}/{len(masks_1st)} -> {save_name}")


    # First time painting the covered area white (for subsequent corrosion treatment)
    white_img = image.copy()
    white_img[covered_mask == 255] = (255, 255, 255)

    # Corrosion/expansion of white_img (original parameters slightly larger: erode_iter=6, dilate_iter=4)
    morphed_mask, discarded_mask = erode_mask(white_img, kernel_size=dict['mask_combine_erode_kernel_size'], erode_iter=dict['mask_combine_erode_erode_iter'], dilate_iter=dict['mask_combine_erode_dilate_iter'])

    first_remain = (morphed_mask > 0).astype(np.uint8) * 255  # black background, remaining white
    remain_rgb = image.copy()
    remain_rgb[first_remain == 0] = (255, 255, 255)
    '''
    remain_rgb_dir = Path(output_dir) / "first_remain_RGB"
    remain_rgb_dir.mkdir(parents=True, exist_ok=True)
    _safe_imwrite(os.path.join(remain_rgb_dir, f"{base}_first_remain_rgb.png"), cv2.cvtColor(remain_rgb, cv2.COLOR_RGB2BGR))
    '''
    print(f"  Finish：saved {len(masks_1st)} instances; first_remain pixels = {cv2.countNonZero(first_remain)}")
    first_remain_bin = morphed_mask
    _safe_imwrite(os.path.join(first_instance_dir, f"{base}_first_remain_bin.png"), first_remain_bin)

    # The union of all mask_first
    union_mask_first = np.zeros((height, width), dtype=np.uint8)
    for mask in mask_first:
        union_mask_first = cv2.bitwise_or(union_mask_first, mask)
    union_mask_first_path = os.path.join(first_instance_dir, f"{base}_first_union_mask.png")
    _safe_imwrite(union_mask_first_path, union_mask_first)

    # The union of mask_first and first_demain_fin
    union_with_remain = cv2.bitwise_or(union_mask_first, first_remain_bin)
    union_with_remain_path = os.path.join(first_instance_dir, f"{base}_all_union_mask.png")
    _safe_imwrite(union_with_remain_path, union_with_remain)
    return  first_remain, mask_first



