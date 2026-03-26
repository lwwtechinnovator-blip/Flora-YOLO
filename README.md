#  Flora-YOLO: A Rose Phenological Stage Grading Model

## 1. Overview
This repository contains the minimal dataset and core scripts required to support the findings presented in the associated manuscript. Due to the large size of the complete dataset (over 8,000 manually annotated rose instances), this package provides a representative subset and the essential configuration files to ensure the reproducibility of our experimental results.

## 2. Dataset Structure
- **data/**: Contains a representative subset of the rose grading dataset.
  - **images/**: Original JPEG images capturing various phenological stages (Degree 0 to Degree 4).
  - **labels/**: Corresponding YOLO-format annotation files (.txt).
- **rose_dataset.yaml**: The data configuration file defining paths and category names for the five grading stages.
- **Flora-YOLO.yaml**: The model architecture configuration file, detailing the integrated C2VGA and modified backbone layers.
- **example_code.py**: A core Python script demonstrating the training pipeline, including hyperparameter settings and data augmentation strategies used in our study.

## 3. Data Description (Metadata)
- **Primary Data**: Raw images and processed labels used for generating performance metrics (mAP, F1-score).
- **Classification Standards**: 
  - Degree 0: Bud
  - Degree 1: Early opening
  - Degree 2: Semi-bloomed
  - Degree 3: Bloomed
  - Degree 4: Fully bloomed
- **Format**: YOLO 1.0 standard (normalized bounding box coordinates).

## 4. Usage Instructions
To reproduce the training environment:
1. Ensure `ultralytics` and `torch` are installed.
2. Configure the paths in `rose_dataset.yaml` to point to the local `data/` folder.
3. Run `python example_code.py` to initiate the training process.

