import os
import json
import pandas as pd
import torch
import torch_npu
from ultralytics import YOLO
from pathlib import Path


def inference(dict, yaml_path, names):
    weights_path = dict['yolo_inference_weights_path']  # 你的权重
    yaml_path = yaml_path
    save_json = dict['yolo_inference_save_json']

    # load model
    model = YOLO(weights_path)
    device = dict['device']
    model.to(device)

    # Reasoning (val mode)
    results = model.val(
        data=yaml_path,
        split="val",
        device=device,
        save=dict['yolo_inference_save'],
        save_txt=dict['yolo_inference_save_txt'],
        save_conf=dict['yolo_inference_save_conf'],
        project=dict['yolo_inference_project'],
        name=dict['yolo_inference_name'],
    )

    # Analyze the saved txt file
    txt_dir = Path(dict['yolo_inference_project']) / Path(dict['yolo_inference_name']) / "labels"

    records = []
    best_records = []

    for file in os.listdir(txt_dir):
        if not file.endswith(".txt"):
            continue

        img_name = file.replace(".txt", ".png")
        file_path = os.path.join(txt_dir, file)

        with open(file_path, "r") as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                cls, xc, yc, w, h, conf = line.strip().split()
                cls = int(cls)
                xc, yc, w, h, conf = map(float, [xc, yc, w, h, conf])

                record = {
                    "img_name": img_name,
                    "typeID": cls,
                    "type_name": names[cls],
                    "conf": conf,
                    "x_center": xc,
                    "y_center": yc,
                    "w": w,
                    "h": h,
                    "area_rate": w * h
                }

                records.append(record)

                # save frst line as best_records
                if i == 0:
                    best_records.append(record)

    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

    df = pd.DataFrame(records)
    df.to_excel(save_json, index=False)
    print(f"The complete result has been exported : {save_json}")

    best_json_path = save_json.replace(".json", "_best.json")
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_records, f, ensure_ascii=False, indent=4)
    print(f"The highest confidence result has been exported: {best_json_path} (total {len(best_records)} )")

    print(f"Total testing results: {len(records)} ,  best.json: {len(best_records)} ")
    return best_json_path

