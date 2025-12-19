import os
import glob
import cv2
import numpy as np
from tqdm import tqdm


def split_disconnected_masks_from_rgb_for_one_image(
    mask,
    save_dir,
    instance_idx,
    base,
    mode='ccomp',               # 'ccomp' (reserved hole)/'external' (only outline)/'tree'
    save_fullsize_mask=True,    # True: Save to the same size as the original image; False: Only save bbox crop
    chain='simple',             # 'simple' or 'none'
    min_area=0,                 # Area filtering (pixels), 0 indicates no filtering
    keep_top_k=None,            # Only retain the top K with the largest area; None means all
    show_progress=False         # Whether to display the progress bar of the mask inside each image
):

    H, W = mask.shape[:2]
    masks_yolo_split = []

    name_root = base
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if np.max(mask) > 1:
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    # Find the outline
    chain_mode = cv2.CHAIN_APPROX_SIMPLE if chain == 'simple' else cv2.CHAIN_APPROX_NONE
    mode_map = {'external': cv2.RETR_EXTERNAL, 'tree': cv2.RETR_TREE, 'ccomp': cv2.RETR_CCOMP}
    retrieval = mode_map.get(mode.lower(), cv2.RETR_CCOMP)
    contours, hierarchy = cv2.findContours(mask, retrieval, chain_mode)
    if hierarchy is not None and len(hierarchy.shape) == 3:
        hierarchy = hierarchy[0]  # Nx4: [next, prev, child, parent]

    def fullsize_mask_and_info(outer_idx):

        c_outer = contours[outer_idx]
        x, y, w, h = cv2.boundingRect(c_outer)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask, [c_outer], -1, 255, thickness=-1)

        hole_ids = []
        if retrieval == cv2.RETR_CCOMP and hierarchy is not None:
            child = hierarchy[outer_idx][2]
            while child != -1:
                hole_ids.append(child)
                child = hierarchy[child][0]
            if hole_ids:
                cv2.drawContours(mask, [contours[i] for i in hole_ids], -1, 0, thickness=-1)

        area_px = int(cv2.countNonZero(mask))
        return mask, (x, y, w, h), hole_ids, area_px, c_outer

    # Outer contour index
    if retrieval == cv2.RETR_CCOMP and hierarchy is not None:
        outer_ids = [i for i, h in enumerate(hierarchy) if h[3] == -1]
    else:
        outer_ids = list(range(len(contours)))

    # Pre statistics and descending order by area (using y and x as stable order for the same area)
    prelim = []
    for oi in outer_ids:
        mask, bbox, hole_ids, area_px, c_outer = fullsize_mask_and_info(oi)
        if area_px >= min_area:
            prelim.append((oi, area_px, bbox, hole_ids, c_outer))
    prelim.sort(key=lambda t: (-t[1], t[2][1], t[2][0]))  # (-area, y, x)

    # Keep the first K
    if isinstance(keep_top_k, int) and keep_top_k is not None and keep_top_k > 0:
        prelim = prelim[:keep_top_k]
    pbar = tqdm(total=len(prelim), desc=f"Processing {name_root}", unit='mask',
                leave=False) if show_progress else None

    # Save and collect information
    for idx, (oi, area_px, (x, y, w, h), hole_ids, c_outer) in enumerate(prelim):
        out_name = f'{name_root}_instance_{instance_idx}_mask_{idx}.png'
        out_path = os.path.join(save_dir, out_name)

        if save_fullsize_mask:
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(mask, [c_outer], -1, 255, thickness=-1)
            if hole_ids:
                cv2.drawContours(mask, [contours[i] for i in hole_ids], -1, 0, thickness=-1)
            cv2.imencode('.png', mask)[1].tofile(out_path)
            masks_yolo_split.append(mask)

        else:
            small = np.zeros((h, w), dtype=np.uint8)
            c_outer_local = c_outer - np.array([[x, y]])
            cv2.drawContours(small, [c_outer_local], -1, 255, thickness=-1)
            if hole_ids:
                holes_local = [contours[i] - np.array([[x, y]]) for i in hole_ids]
                cv2.drawContours(small, holes_local, -1, 0, thickness=-1)
            cv2.imencode('.png', small)[1].tofile(out_path)
            masks_yolo_split.append(small)

        if pbar: pbar.update(1)

    if pbar: pbar.close()


def process_folder(
    dict,
    masks,               # Source Image Folder
    output_dir,          # Output mask large folder
    base
):
    binary_dir = dict['mask_split_binary_dir']
    if binary_dir is not None:
        os.makedirs(binary_dir, exist_ok=True)

    for instance_idx, mask in enumerate(masks) :
        split_disconnected_masks_from_rgb_for_one_image(
            mask=mask,
            save_dir=output_dir,
            instance_idx = instance_idx,
            base=base,
            mode=dict['mask_split_mode'],
            save_fullsize_mask=dict['mask_split_save_fullsize_mask'],
            chain=dict['mask_split_chain'],
            min_area=dict['mask_split_min_area'],
            keep_top_k=dict['mask_split_keep_top_k'],
            show_progress=dict['mask_split_show_progress'],
        )

