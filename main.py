from modules.sam_first import process_first_stage
from modules.mask_combine import process_combine
from modules.sam_remain import process_others_stage
from modules.init import dict_init
from modules.mask_split import process_folder
from modules.mask_cut import mask_cut
from modules.yolo_yaml import yaml_generate
from modules.yolo_inference import inference
from modules.sam_second import process_second_stage
from pathlib import Path
import os
import glob
import torch
import torch_npu
import cv2
import numpy as np

if __name__ == "__main__":

    if torch.npu.is_available():
        device = torch.device("npu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    dict = dict_init()
    input_dir = dict['input_dir']
    output_dir = dict['output_dir']

    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    mask_split_output_dir = Path(output_dir) / "mask_split"
    os.makedirs(mask_split_output_dir, exist_ok=True)
    mask_cut_output_dir = Path(output_dir) / "mask_cut"
    os.makedirs(mask_cut_output_dir, exist_ok=True)
    yaml_path = str(Path(output_dir) / "dataset_infer.yaml")

    valid_extensions = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG", ".bmp"]
    img_list = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                img_list.append(os.path.join(root, file))
    img_list = sorted(set(img_list))

    print(f"find {len(img_list)} image files")
    input_dir_raw = Path(output_dir) / "first_instance_bin"
    input_dir_raw.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(img_list):
        print("Executing module1")
        masks_1st, image, base = process_first_stage(dict, img_path)
        print("Executing module2")
        first_remain, mask_first = process_combine(dict, masks_1st,image, base)
        print("Executing module3")
        mask_second = process_second_stage(dict, image, mask_first,base)
        print("Executing module4")
        mask_others = process_others_stage(dict, first_remain, image, base)
        print("Executing module5")
        all_masks = mask_others  + mask_second

        process_folder(dict, all_masks, mask_split_output_dir, base)
    print("Executing module6")
    mask_cut(dict, input_dir, mask_split_output_dir, mask_cut_output_dir)
    print("Executing module7")
    yaml_generate(mask_cut_output_dir, yaml_path, names)
    print("Executing module8")
    inference(dict, yaml_path= yaml_path, names)

