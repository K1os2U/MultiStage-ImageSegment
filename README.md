# SAM Segmentation and YOLO Instance Inference Project
## Project Overview
This project implements a complete pipeline for image segmentation using SAM (Segment Anything Model) and instance inference with YOLO. The project includes multiple modular components that perform multi-stage segmentation, instance integration, secondary segmentation, remaining region processing, and final instance recognition and classification with YOLO models.

When separating images in this project, SAM is first used to perform initial segmentation of the image subject, calculate the IOU separating each image, merge adjacent images, and obtain multiple large blocks of separated images and unsegmented areas. Perform more rounds of separation on multiple separated images and unsegmented area images separately. Adjust SAM parameters in each link, and add functions such as the identified area shielding, edge corrosion, and small area contour filtering. Ensure that the final output image instance meets the required accuracy, size, and overlap degree for the target. This project can freely adjust the segmentation effect of each process in multiple rounds, and various post-processing algorithms are added to ensure optimization effect, which can maximize the absence of undivided areas
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
'yolo_model_path': "./models/yolo/yolov11.pt",         # YOLO model weights
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
'sam_first_crop_n_layers': 1,
'sam_first_crop_nms_thresh': 0.7,
'sam_first_crop_overlap_ratio': 512 / 1500,
'sam_first_min_mask_region_area': 5000,
'sam_first_output_mode': 'binary_mask',
```
**MASK PROCESSING PARAMETERS**
```
'mask_combine_iou_filter_thresh': 0.9,      # IOU filter threshold
'mask_combine_min_area_ratio': 0.002,       # Minimum area filtration ratio
'mask_combine_contain_thresh': 0.9,         # Image inclusion ratio
'mask_combine_erode_kernel_size': 3,         # Erosion related parameters
'mask_combine_erode_erode_iter': 6,
'mask_combine_erode_dilate_iter': 4,
```
**MASK SPLITING PARAMETERS**
```
'mask_split_mode': 'ccomp',              # 'ccomp' (reserved hole)/'external' (only outline)/'tree'
'mask_split_save_fullsize_mask': True,    # True: Save to the same size as the original image; False: Only save bbox crop
'mask_split_chain':'simple',             # 'simple' or 'none'
'mask_split_min_area': 0,                 # Area filtering (pixels), 0 indicates no filtering
'mask_split_keep_top_k': None,            # Only retain the top K with the largest area; None means all
'mask_split_show_progress': False,         # Whether to display the progress bar of the mask inside each image
```
**YOLO INFERENCE PARAMETERS**
```
'yolo_inference_save_json': "./runs/inference.json",  # Output JSON path
'yolo_inference_name': "inference",                   # Inference name
'yolo_inference_project': "./runs/detect",            # Inference save path
'yolo_inference_num_classes' : 19                     # Inference category quantity
```
*For the parameters in init.by, it is necessary to modify them to the data source, model input, and output path. The project includes operations such as mask merging, corrosion, small area filtering, and already identified area filtering. It has multiple adjustable parameters, and both Sam and YOLO related parameters can be modified. These optional parameters can be customized for better task processing*
## Main Functional Modules
The project file mainly includes three parts: 1. Call the main.py function throughout the entire process; 2. A single module calls the function module_XX_XX.py; 3. Implement the code for a single module in the directory "./modules"
### Module 1: Initial Segmentation (module_sam_first.py)
Function: Perform initial segmentation using SAM model

Core Function: process_first_stage(dict, img_path)

Input: Raw image (supports .jpg, .png, .jpeg, .bmp)

Output: Initial segmentation masks

Key Parameters: See init.pyfor SAM configuration
### Module 2: Mask Integration (module_mask_combine.py)

Function: Integrate and optimize initial segmentation masks

Core Function: process_first_stage(dict, img_path)

Processing: Merge overlapping masks, separate adhesive regions, fill holes

Key Parameters: IOU threshold, erosion kernel size
### Module 3: Secondary Segmentation (module_sam_second.py)

Function: Fine-grained segmentation of integrated mask regions

Core Function: process_second_stage(dict, image, mask_first, base)

Features: Padding processing for better edge segmentation
### Module 4: Remaining Region Processing (module_sam_remain.py)

Function: Process unrecognized regions from previous steps

Core Function: process_others_stage(dict, first_remain, image, base)

Use Case: Complex background and small object recognition
### Module 5: Mask Separation (module_mask_split.py)

Function: Separate final segmentation results into independent instances

Core Function: process_folder(dict, all_masks, mask_split_output_dir, base)

Output: Mask files at original image size
### Module 6-9: YOLO Inference (module_yolo_inference.py)

Function: Object detection and classification of segmented instances

Core Function: process_folder(dict, all_masks, mask_split_output_dir, base)
### Output:

Classification results JSON file

Labeled mask images

Inference process files
## Quick Start (main function)
Need to adjust the input and output in init.py, and put the necessary data into the directory (including images, SAM models, Yolo models). In the YOLO recognition section, it is important to specify the category names of the classification by yourself(Please note that the JSON reading path and YOLO detection result path must be consistent). 

After ensuring that the input and output data and path configuration are correct, the main function can be used for multi round separation and recognition of the complete process.
```
python main.py
```
Get the output and midway processing files at the end of the process
