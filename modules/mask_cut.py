import copy
import os
from pathlib import Path
import glob
import cv2
import numpy as np
import math


def mask_cut(dict, input_dir, mask_split_output_dir, mask_cut_output_dir):
    padding_ratio = dict['mask_cut_padding_ratio']

    # Establish a list of original image paths
    exts = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG","bmp"]
    img_list = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in exts):
                img_list.append(os.path.join(root, file))
    img_list = sorted(img_list)
    img_base_list = [os.path.splitext(os.path.basename(f))[0] for f in img_list]

    # Establish a mask path list
    mask_list = sorted(glob.glob(os.path.join(mask_split_output_dir, "*")))

    for file_i, img_path in enumerate(img_list):
        # load the original image
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
        img_h, img_w = img.shape[:2]

        base_name = img_base_list[file_i]
        # Find all masks corresponding to the current original image
        mask_idx = [i for i in range(len(mask_list))
                    if os.path.basename(mask_list[i]).startswith(base_name + "_instance")]

        for mask_i in mask_idx:
            mask_path = mask_list[mask_i]
            mask_img = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), 1)
            mask_img_resize = cv2.resize(mask_img, (img_w, img_h))

            # save
            mask_name = os.path.splitext(os.path.basename(mask_path))[0] + "_RGB" + os.path.splitext(mask_path)[1]

            instance = copy.deepcopy(img)
            instance[np.all(mask_img_resize == (0, 0, 0), axis=-1)] = (255, 255, 255)

            bg = (mask_img_resize[..., 0] == 0) & (mask_img_resize[..., 1] == 0) & (mask_img_resize[..., 2] == 0)
            bw_img = np.where(bg, 0, 255).astype(np.uint8)

            x = np.sum(bw_img, 0)
            x0 = list(np.nonzero(x)[0])[0]
            x1 = list(np.nonzero(x)[0])[-1]
            y = np.sum(bw_img, 1)
            y0 = list(np.nonzero(y)[0])[0]
            y1 = list(np.nonzero(y)[0])[-1]

            ins_w = x1 - x0 + 1
            ins_h = y1 - y0 + 1

            img_w_new = ins_w + 2 * math.ceil(ins_w * padding_ratio)
            img_h_new = ins_h + 2 * math.ceil(ins_h * padding_ratio)

            img_new = np.zeros((img_h_new, img_w_new, 3), dtype=np.uint8) + 255
            img_new[math.ceil(ins_h * padding_ratio):math.ceil(ins_h * padding_ratio) + ins_h,
            math.ceil(ins_w * padding_ratio):math.ceil(ins_w * padding_ratio) + ins_w, :] \
                = instance[y0:y1 + 1, x0:x1 + 1, :]

            # Save file with the same name as mask file
            out_path = os.path.join(mask_cut_output_dir, mask_name)
            cv2.imencode('.png', img_new)[1].tofile(out_path)


