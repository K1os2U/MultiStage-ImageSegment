from models.mask_cut import mask_cut
from models.yolo_yaml import yaml_generate
from models.yolo_inference import inference
from models.init_2 import dict_init
from models.results import rename_files
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
    mask_split_output_dir = "./XXXX"
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    output_dir = dict['output_dir']
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mask_cut_output_dir = Path(output_dir) / "mask_cut"
    os.makedirs(mask_cut_output_dir, exist_ok=True)
    yaml_path = str(Path(output_dir) / "dataset_infer.yaml")
    results_output_dir = Path(output_dir) / "results"
    os.makedirs(results_output_dir, exist_ok=True)

    print("module 6")
    mask_cut(dict, input_dir, mask_split_output_dir, mask_cut_output_dir)
    print("module 7")
    yaml_generate(mask_cut_output_dir, yaml_path)
    print("module 8")
    best_json_path = inference(dict, yaml_path= yaml_path)
    print("module 9")
    rename_files(best_json_path, mask_split_output_dir, results_output_dir)

