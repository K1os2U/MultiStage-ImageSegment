from modules.sam_first import process_first_stage
from modules.init import dict_init
import os
import cv2
import glob
from pathlib import Path

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
    output_dir = dict['output_dir']
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    valid_extensions = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG", ".bmp"]

    img_list = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                img_list.append(os.path.join(root, file))
    img_list = sorted(set(img_list))

    print(f"find  {len(img_list)} image files")

    for idx, img_path in enumerate(img_list):
        print("module 1")
        masks_1st, image, base = process_first_stage(dict, img_path)

        for j, mask in enumerate(masks_1st):
            mask_filename = f"{base}_first_mask_{j}.png"
            mask_output_path = os.path.join(output_dir, mask_filename)
            _safe_imwrite(mask_output_path, mask)
        print(f"  get {len(masks_1st)} masks")

