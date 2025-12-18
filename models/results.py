import os
import json
import shutil

def rename_files(json_path, src_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Establish a source file name index table
    src_files_lower = {f.lower(): f for f in os.listdir(src_dir)}
    # Record the cumulative number of each prefix_type_ {category_id}
    instance_counters = {}

    count_exist = 0
    count_missing = 0
    count_copied = 0

    for file_info in data:
        original_name = os.path.basename(file_info.get("img_name", "")).strip()
        category_id = file_info.get("typeID", "")

        if not original_name:
            print(f"JSON entry is missing fields：{file_info}")
            continue
        if not str(category_id).isdigit():
            print(f"Category ID exception：{category_id}（file：{original_name}）")
            continue

        # Match source files
        if original_name.endswith("_RGB.png"):
            src_name = original_name.replace("_RGB.png", ".png")
        else:
            src_name = original_name
        if "_instance_" not in src_name:
            print(f"The file name does not comply with the specifications : {src_name}")
            continue

        prefix = src_name.split("_instance_")[0]

        group_key = f"{prefix}_type_{category_id}"
        if group_key not in instance_counters:
            instance_counters[group_key] = 0
        else:
            instance_counters[group_key] += 1

        instance_idx = instance_counters[group_key]

        # Construct a new file name
        new_name = f"{group_key}_instance_{instance_idx}.png"

        # Check if the source file exists
        name_lower = src_name.lower()
        if name_lower in src_files_lower:
            src_path = os.path.join(src_dir, src_files_lower[name_lower])
            dst_path = os.path.join(output_dir, new_name)
            shutil.copy2(src_path, dst_path)
            count_copied += 1
            count_exist += 1
        else:
            print(f"can not find file: {src_name}")
            count_missing += 1

    print(f"output: {output_dir}")


