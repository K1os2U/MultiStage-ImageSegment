y# SAM Segmentation and YOLO Instance Inference Project
## Project Overview
This project implements a complete pipeline for image segmentation using SAM (Segment Anything Model) and instance inference with YOLO. The project includes multiple modular components that perform multi-stage segmentation, instance integration, secondary segmentation, remaining region processing, and final instance recognition and classification with YOLO models.
./runs/detect/train/weights/best.pt
./SAM/sam_vit_h_4b8939.pth
../inputs/sam_remain_class_img_val_19_3_0.01_split"
> project_root
>> inputs                         # Input directories
>>> datasets                    # Raw input images for inference
>>>> image1.jpg
>>>> image2.png
>>>> ...
>> SAM                         # SAM model directory
>>> sam_vit_b_01ec64.pth     # SAM model weights
>>>> ...                     # Other SAM model files
>> runs
>>> detect
>>>> train
>>>>> weight
>>>>>> yolov11.ptol
