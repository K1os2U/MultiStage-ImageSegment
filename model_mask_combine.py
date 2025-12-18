from models.mask_combine import process_combine
from models.init import dict_init
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
    mask_dir = "./XXX"
    output_first = dict['output_dir']
    output_remain = dict['output_dir']
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(mask_dir).mkdir(parents=True, exist_ok=True)
    Path(output_first).mkdir(parents=True, exist_ok=True)
    Path(output_remain).mkdir(parents=True, exist_ok=True)

    valid_extensions = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG", ".bmp"]

    img_list = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                img_list.append(os.path.join(root, file))
    img_list = sorted(set(img_list))

    print(f"find {len(img_list)} image file")

    for idx, img_path in enumerate(img_list):
        image_bgr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        base = os.path.splitext(os.path.basename(img_path))[0]

        mask_pattern = os.path.join(mask_dir, f"{base}_first*.png")
        mask_paths = sorted(glob.glob(mask_pattern))
        masks_1st = []
        for m_path in mask_paths:
            mask_img = cv2.imdecode(np.fromfile(m_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            _, mask_bin = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
            masks_1st.append(mask_bin)
        if len(masks_1st) == 0:
            print(f" No corresponding mask file found: {base}_first *.png， Skip this image")
            continue

        print("module 2")
        first_remain, mask_first = process_combine(dict, masks_1st, image, base)

        # Save output as images of various masks
        for j, mask in enumerate(mask_first):
            mask_filename = f"{base}_combine_mask_{j}.png"
            mask_output_path = os.path.join(output_first, mask_filename)
            _safe_imwrite(mask_output_path, mask)
        remain_filename = f"{base}_remain_mask.png"
        mask_output_path = os.path.join(output_remain, remain_filename)
        _safe_imwrite(mask_output_path, first_remain)
        print(f" get {len(mask_first)} masks and 1 remain_mask")

