import os
import yaml

def yaml_generate(dataset_path, yaml_path,names):
    dataset_path = os.path.abspath(str(dataset_path))


    yaml_dict = {
        "path": dataset_path,
        "train": ".",
        "val": ".",
        "nc": len(names),
        "names": names
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, allow_unicode=True)

    print(f"YAML file: {os.path.abspath(yaml_path)}")
    print("data.path =", yaml_dict['path'])
