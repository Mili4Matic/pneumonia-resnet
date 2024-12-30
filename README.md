# Pneumonia Detection with ResNet50

This project demonstrates how to detect pneumonia from chest X-ray images using a Transfer Learning approach with ResNet50. 

## Features
- Two-phase training: head-only, then fine-tuning.
- Accuracy and loss plots with Matplotlib.
- Inference script for single-image prediction.

## Repository Structure
pneumonia-resnet/
├── README.md
├── .gitignore
├── requirements.txt
├── pneumonia_train.py (training script)
├── pneumonia_infer.py (inference script)
├── data/
    │ └── chest_xray/ │
          ├── train/ │
          ├── val/ │
          └── test/
    └── models/
          └── pneumonia_resnet50_tf25.h5
