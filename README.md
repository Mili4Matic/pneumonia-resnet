# Pneumonia Detection with ResNet50

This project demonstrates how to detect pneumonia from chest X-ray images using a Transfer Learning approach with ResNet50. 

> **Note**: The `data/chest_xray` folder may be too large for GitHub if it contains the entire dataset, uploaded images are samples of the dataset. Instead, download it from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia.
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
│     ├── train/ │  
│     ├── val/ │  
│     └── test/  
└── models/  
│      └── pneumonia_resnet50_tf25.h5  

        
Usage

    Download/Place Data
        Download chest_xray from [Kaggle Link] into data/chest_xray.
        Ensure you have train/, val/, test/ subdirectories.

    Train the model

python pneumonia_train.py

    This will produce pneumonia_resnet50_tf25.h5 in models/ (or current folder).

Run Inference

    python pneumonia_infer.py /path/to/chest_xray/test/NORMAL/IM-0001.jpg

        Outputs whether it’s pneumonia or normal with a probability.

Results

    With 10 epochs on head + 10 epochs fine-tuning, we achieved ~85% accuracy (example).
    For best results, try adjusting epochs, data augmentation, and the number of unfrozen layers.
