# SAM Segmentation and YOLO Instance Inference Project
## Project Overview
This project implements a complete pipeline for image segmentation using SAM (Segment Anything Model) and instance inference with YOLO. The project includes multiple modular components that perform multi-stage segmentation, instance integration, secondary segmentation, remaining region processing, and final instance recognition and classification with YOLO models.
## Prerequisites and Setup
Before running the project, ensure you have the following file structure:

├── inputs/                          # Input directories

│   ├── datasets/                    # Raw input images for inference

│   │   ├── image1.jpg

│   │   ├── image2.png

│   │   └── ...

├── models/                          # Model files

│   ├── sam/                         # SAM model directory

│   │   ├── sam_vit_b_01ec64.pth     # SAM model weights

│   │   └── ...                     # Other SAM model files

│   ├── yolo/                      # YOLO model directory

│   │   ├── yolov11.pt              # YOLO model weights

│   │   └── ...

## Configuration (models/init.py)
**PATHS AND DIRECTORIES** 
```
'input_dir': "../inputs/datasets",           # Path to input images
'output_dir': "./output/results",           # Path for output results
'models_dir': "./models",                   # Path to model directory
```
**SAM MODEL PATHS**
```
'sam_model_path': "./models/sam/sam_vit_b_01ec64.pth",  # SAM model weights
'sam_model_type': "vit_b",                               # SAM model type
```
**YOLO MODEL PATHS** 
```
'yolo_model_path': "./models/yolov5/yolov5s.pt",         # YOLO model weights
'yolo_data_yaml': "./models/yolov5/data/coco128.yaml",   # YOLO data config
```
**DEVICE CONFIGURATION**
```
'device': "cuda:0",                          # Use "cpu", "cuda:0", or "npu:0"
```
**SAM PARAMETERS**
```
'sam_first_points_per_side': 32,            # Points per side (16-64)
'sam_first_pred_iou_thresh': 0.9,           # IOU threshold (0.75-0.95)
'sam_first_stability_score_thresh': 0.9,    # Stability score threshold
```
**MASK PROCESSING PARAMETERS**
```
'mask_combine_iou_filter_thresh': 0.5,      # IOU filter threshold
'mask_combine_erode_kernel_size': 3,         # Erosion kernel size
```
**YOLO INFERENCE PARAMETERS**
```
'yolo_inference_save_json': "./runs/inference.json",  # Output JSON path
'yolo_inference_name': "inference",                   # Inference name
'yolo_conf_threshold': 0.25,                          # Confidence threshold
'yolo_iou_threshold': 0.45                            # IOU threshold
```
*For the parameters in init.by, it is necessary to modify them to the data source, model input, and output path. The project includes operations such as mask merging, corrosion, small area filtering, and already identified area filtering. It has multiple adjustable parameters, and both Sam and YOLO related parameters can be modified. These optional parameters can be customized for better task processing*
## Main Functional Modules
### Module 1: Initial Segmentation (models_sam_first.py)
Function: Perform initial segmentation using SAM model

Core Function: process_first_stage(dict, img_path)

Input: Raw image (supports .jpg, .png, .jpeg, .bmp)

Output: Initial segmentation masks

Key Parameters: See init.pyfor SAM configuration
### Module 2: Mask Integration (models_mask_combine.py)

Function: Integrate and optimize initial segmentation masks

Core Function: process_first_stage(dict, img_path)

Processing: Merge overlapping masks, separate adhesive regions, fill holes

Key Parameters: IOU threshold, erosion kernel size
### Module 3: Secondary Segmentation (models_sam_second.py)

Function: Fine-grained segmentation of integrated mask regions

Core Function: process_second_stage(dict, image, mask_first, base)

Features: Padding processing for better edge segmentation
### Module 4: Remaining Region Processing (models_sam_remain.py)

Function: Process unrecognized regions from previous steps

Core Function: process_others_stage(dict, first_remain, image, base)

Use Case: Complex background and small object recognition
### Module 5: Mask Separation (model_mask_split.py)

Function: Separate final segmentation results into independent instances

Core Function: process_folder(dict, all_masks, mask_split_output_dir, base)

Output: Mask files at original image size
### Module 6-9: YOLO Inference (model_yolo_inference.py)

Function: Object detection and classification of segmented instances

Core Function: process_folder(dict, all_masks, mask_split_output_dir, base)
### Output:

Classification results JSON file

Labeled mask images

Inference process files
## Quick Start
Adjust the input and output in init.py, and put the necessary data into the directory (including images, SAM models, Yolo models) to start multiple rounds of separation and identification.
```
python main.py
```
Get the output and midway processing files at the end of the process
