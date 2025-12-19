import os
import sys
import cv2
import numpy as np
import math
import torch
import torch_npu
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator



def _safe_imwrite(path, img):
    ext = os.path.splitext(path)[1]
    if ext == "":
        ext = ".png"
        path = path + ext
    _, buf = cv2.imencode(ext, img)
    buf.tofile(path)

def extract_remain(base_mask, sub_mask):
    return cv2.bitwise_and(base_mask, cv2.bitwise_not(sub_mask))

def compute_area(m):
    return np.count_nonzero(m["segmentation"])

def apply_morphology(mask, cfg, H, W):
    short_side = min(H, W)
    if cfg["sam_2nd_kernel_ratio"] > 0:
        ksize = max(1, int(short_side * cfg["sam_2nd_kernel_ratio"]))
        if ksize % 2 == 0:
            ksize += 1
    else:
        ksize = 3
    kernel = np.ones((ksize, ksize), np.uint8)
    out = mask.copy()
    if cfg["sam_2nd_erode_iter"] > 0:
        out = cv2.erode(out, kernel, iterations=cfg["sam_2nd_erode_iter"])
    if cfg["sam_2nd_dilate_iter"] > 0:
        out = cv2.dilate(out, kernel, iterations=cfg["sam_2nd_dilate_iter"])
    return out

# Initialize SAM
def init_sam(cfg):
    device = cfg['device']
    sam = sam_model_registry[cfg["sam_2nd_model_type"]](checkpoint=cfg["sam_2nd_model_path"])
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=cfg["sam_2nd_points_per_side"],
        pred_iou_thresh=cfg["sam_2nd_pred_iou_thresh"],
        stability_score_thresh=cfg["sam_2nd_stability_score_thresh"],
        crop_n_layers=cfg["sam_2nd_crop_n_layers"],
        crop_nms_thresh=cfg["sam_2nd_crop_nms_thresh"],
        crop_overlap_ratio=cfg["sam_2nd_crop_overlap_ratio"],
        min_mask_region_area=cfg["sam_2nd_min_mask_region_area"],
        output_mode=cfg["sam_2nd_output_mode"]
    )
    return mask_generator

# Initial instance processing
def process_first_instances(cfg, ori_img, mask_list):
    results = []
    h_img, w_img = ori_img.shape[:2]

    for idx, mask_img in enumerate(mask_list, 1):
        ys, xs = np.where(mask_img > 0)
        if ys.size == 0 or xs.size == 0:
            continue

        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        ins_w, ins_h = x1 - x0 + 1, y1 - y0 + 1

        ins_crop = ori_img[y0:y1+1, x0:x1+1, :].copy()
        mask_crop = mask_img[y0:y1+1, x0:x1+1].copy()

        ins_rgb = np.ones_like(ins_crop, dtype=np.uint8) * 255
        mask_bool = (mask_crop > 0)
        ins_rgb[mask_bool] = ins_crop[mask_bool]

        padding_size = math.ceil(max(ins_h, ins_w) * cfg["sam_2nd_padding_ratio"])
        new_h, new_w = ins_h + 2 * padding_size, ins_w + 2 * padding_size

        canvas_rgb = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255
        canvas_rgb[padding_size:padding_size+ins_h, padding_size:padding_size+ins_w, :] = ins_rgb

        results.append({
            "instance_id": idx,
            "canvas_rgb": canvas_rgb,
            "bbox": (x0, y0, ins_w, ins_h, padding_size),
            "H": h_img,
            "W": w_img
        })

    return results


def run_recursive_structure_split(cfg, ori_img, instance_data, mask_generator):

    H, W = ori_img.shape[:2]
    canvas_rgb = instance_data["canvas_rgb"]
    h_can, w_can = canvas_rgb.shape[:2]
    max_side = max(h_can, w_can)
    size_rate = 1.0
    resize_max_side = cfg["sam_first_resize_max_side"]
    if max_side > resize_max_side:
        size_rate = resize_max_side / max_side

    new_h, new_w = max(1, int(h_can * size_rate)), max(1, int(w_can * size_rate))
    resized_img = cv2.resize(canvas_rgb, (new_w, new_h))

    raw_masks = mask_generator.generate(resized_img)

    # Resize back to canvas size
    for m in raw_masks:
        m["segmentation"] = cv2.resize(
            m["segmentation"].astype(np.uint8),
            (w_can, h_can),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    # SAM seg
    bg_thresh = 10
    origin_bin = np.any(canvas_rgb < (255 - bg_thresh), axis=2).astype(np.uint8)
    origin_area = np.count_nonzero(origin_bin)

    sub_masks = []
    for m in raw_masks:
        seg = m["segmentation"].astype(np.uint8)
        if compute_area(m) == 0:
            continue
        masked_img = np.full_like(canvas_rgb, 255)
        masked_img[seg > 0] = canvas_rgb[seg > 0]
        sub_area = np.sum(~np.all(masked_img == [255, 255, 255], axis=-1))
        ratio = sub_area / origin_area if origin_area > 0 else 0
        if ratio <= cfg["sam_2nd_min_area_ratio"]:
            continue
        sub_masks.append(m)

    sub_masks.sort(key=lambda m: compute_area(m), reverse=True)
    existing_structures = []
    counter = 1

    if not sub_masks:
        return {"instance_id": instance_data["instance_id"], "binary_masks": []}

    # Recursive structure segmentation
    mask_1 = sub_masks[0]["segmentation"].astype(np.uint8)
    remain_1 = extract_remain(origin_bin, mask_1)
    existing_structures.append({"name": f"mask_{counter}", "binary": mask_1})
    counter += 1
    existing_structures.append({"name": f"remain_{counter-1}", "binary": remain_1})

    for i in range(1, len(sub_masks)):
        mask_bin = sub_masks[i]["segmentation"].astype(np.uint8)
        area = np.count_nonzero(mask_bin)
        best_match, best_overlap = None, 0
        for m in existing_structures:
            inter = np.logical_and(mask_bin == 1, m["binary"] == 1).sum()
            overlap = inter / area if area > 0 else 0
            if overlap > best_overlap:
                best_overlap, best_match = overlap, m
        if best_overlap >= cfg["sam_2nd_overlap_thresh"]:
            remain = extract_remain(best_match["binary"], mask_bin)
            best_match["binary"] = remain
            existing_structures.append({"name": f"mask_{counter}", "binary": mask_bin})
            counter += 1

    # Binary mask output (restored to original image size)
    binary_masks = []
    a = 0
    for s in existing_structures:
        a += 1
        bin_mask = (s["binary"] > 0).astype(np.uint8)
        bin_mask = apply_morphology(bin_mask, cfg, H, W)

        area = np.count_nonzero(bin_mask)
        if area / origin_area < cfg["sam_2nd_min_area_ratio"]:
            continue

        # Restore to original image size
        x0, y0, ins_w, ins_h, pad = instance_data["bbox"]
        crop_h, crop_w = ins_h, ins_w
        H_img, W_img = instance_data["H"], instance_data["W"]

        # remove padding
        mask_unpad = bin_mask[pad:pad + crop_h, pad:pad + crop_w]

        # Restore the original image size
        full_mask = np.zeros((H_img, W_img), dtype=np.uint8)
        y1 = min(y0 + mask_unpad.shape[0], H_img)
        x1 = min(x0 + mask_unpad.shape[1], W_img)
        full_mask[y0:y1, x0:x1] = mask_unpad[:y1 - y0, :x1 - x0]

        binary_masks.append(full_mask)

    return binary_masks

def merge_and_save_masks(mask_list, output_dir):
    if not mask_list:
        print("Warning: The mask list is empty")
        return

    # Initialize all zero mask
    merged_mask = np.zeros_like(mask_list[0], dtype=np.uint8)
    for mask in mask_list:
        merged_mask = cv2.bitwise_or(merged_mask, mask)

    # save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "merged_mask.png")
    cv2.imwrite(output_path, merged_mask * 255)

    return merged_mask

def process_second_stage(cfg, image, first_mask_list, base):
    mask_generator = init_sam(cfg)

    # Initial instance processing
    first_stage_results = process_first_instances(cfg, image, first_mask_list)

    # Secondary segmentation
    all_second_stage = []
    for inst in first_stage_results:
        result = run_recursive_structure_split(cfg, image, inst, mask_generator)
        all_second_stage.extend(result)
    output_dir = cfg['output_dir']
    first_instance_dir = Path(output_dir) / "second_instance_bin"
    first_instance_dir.mkdir(parents=True, exist_ok=True)
    merged_mask = merge_and_save_masks(all_second_stage, first_instance_dir)

    return all_second_stage


