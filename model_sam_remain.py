from models.sam_remain import process_others_stage
from models.init_2 import dict_init
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
    output_remain = dict['output_dir']
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(mask_dir).mkdir(parents=True, exist_ok=True)
    Path(output_remain).mkdir(parents=True, exist_ok=True)

    valid_extensions = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG", ".bmp"]
    img_list = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                img_list.append(os.path.join(root, file))
    img_list = sorted(set(img_list))
    print(f"find {len(img_list)} image files")

    for idx, img_path in enumerate(img_list):
        image_bgr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        base = os.path.splitext(os.path.basename(img_path))[0]

        mask_pattern = os.path.join(mask_dir, f"{base}_remain*.png")
        mask_paths = sorted(glob.glob(mask_pattern))
        first_remain = cv2.imdecode(np.fromfile(mask_paths[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        print("module 4")
        mask_others = process_others_stage(dict, first_remain, image, base)

        for j, mask in enumerate(mask_others):
            mask_filename = f"{base}_other_mask_{j}.png"
            mask_output_path = os.path.join(output_remain, mask_filename)
            _safe_imwrite(mask_output_path, mask)
        print(f"  get {len(mask_others)} mask")

