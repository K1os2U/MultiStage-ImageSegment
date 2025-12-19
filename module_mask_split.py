from modules.mask_split import process_folder
from modules.init_2 import dict_init
import os
import cv2
import glob
from pathlib import Path
import numpy as np

def _safe_imwrite(path, img):
    ext = os.path.splitext(path)[1]
    if ext == "":
        ext = ".png"
        path = path + ext
    _, buf = cv2.imencode(ext, img)
    buf.tofile(path)

if __name__ == "__main__":
    dict = dict_init()
    input_dir = dict['input_dir']
    mask_dir_second = "./XXX"
    mask_dir_other = "./XXX"
    output_dir = dict['output_dir']
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(mask_dir_second).mkdir(parents=True, exist_ok=True)
    Path(mask_dir_other).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mask_split_output_dir = Path(output_dir) / "mask_split"
    os.makedirs(mask_split_output_dir, exist_ok=True)

    valid_extensions = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG", ".bmp"]

    img_list = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                img_list.append(os.path.join(root, file))
    img_list = sorted(set(img_list))

    print(f"find {len(img_list)} image files")

    for idx, img_path in enumerate(img_list):
        base = os.path.splitext(os.path.basename(img_path))[0]

        mask_pattern = os.path.join(mask_dir_second, f"{base}_second*.png")
        mask_paths = sorted(glob.glob(mask_pattern))
        mask_second = []
        for m_path in mask_paths:
            mask_img = cv2.imdecode(np.fromfile(m_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            mask_second.append(mask_img)
        mask_other = []
        for m_path in mask_paths:
            mask_img = cv2.imdecode(np.fromfile(m_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            mask_other.append(mask_img)
        all_masks = mask_second + mask_other
        if len(all_masks) == 0:
            print(f" No corresponding mask file found: {base}_combine *.png， Skip this image ")
            continue

        print("module 5")
        process_folder(dict, all_masks, mask_split_output_dir, base)


