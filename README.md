# Pneumonia Detection with ResNet50

This project demonstrates how to detect pneumonia from chest X-ray images using a Transfer Learning approach with ResNet50. 

> **Note**: The `data/chest_xray` folder may be too large for GitHub if it contains the entire dataset. Instead, download it from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia.
> 
## Features
- Two-phase training: head-only, then fine-tuning.
- Accuracy and loss plots with Matplotlib.
- Inference script for single-image prediction.

## How to run test
python pneumonia_test.py /path/to/image.jpg
## Repository Structure  
pneumonia-resnet/  
├── README.md  
├── .gitignore  
├── requirements.txt  
├── pneumonia_train.py (training script)  
├── pneumonia_test.py (inference test)  
├── data/  
│ └── chest_xray/ │  
      ├── train/ │  
      ├── val/ │  
      └── test/  
└── models/  
      └── pneumonia_resnet50_tf25.h5  
 
