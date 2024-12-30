#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# -------------------------------------------------------------------------
# Path to the saved model
# -------------------------------------------------------------------------
MODEL_PATH = "/data/models/pneumonia_resnet50.h5"

# -------------------------------------------------------------------------
# Image processing parameters
# -------------------------------------------------------------------------
IMG_SIZE = (224, 224)

# -------------------------------------------------------------------------
# Function: load_and_preprocess
# -------------------------------------------------------------------------
def load_and_preprocess(img_path):
    """
    Loads an image from disk, resizes it to IMG_SIZE, 
    converts it to a float32 array, and normalizes its pixel values 
    for the model's expected input.
    """
    # Load image at the specified target size
    img = image.load_img(img_path, target_size=IMG_SIZE)
    
    x = image.img_to_array(img)

    # Scale pixel values from [0, 255] to [0, 1]
    x = x / 255.0

    # Expand dimensions to (1, 224, 224, 3) for prediction
    x = np.expand_dims(x, axis=0)
    return x

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    # Check if an image path was provided
    if len(sys.argv) < 2:
        print("Usage: python pneumonia_test.py <image_path>")
        sys.exit(1)

    # Read the image path argument
    img_path = sys.argv[1]

    # Load the pre-trained model
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Preprocess the input image
    x = load_and_preprocess(img_path)

    # Make a prediction
    #    The model returns a single value in the range [0, 1].
    #    If it's near 1 => higher probability of "PNEUMONIA",
    #    if it's near 0 => "NORMAL"
    pred = model.predict(x)[0][0]

    # Convert the raw prediction into a percentage
    prob_neumonia = pred * 100.0

    # Determine the predicted label
    if pred >= 0.5:
        label = "PNEUMONIA"
    else:
        label = "NORMAL"

    # Print the results
    print(f"Image: {img_path}")
    print(f"Prediction: {label}")
    print(f"Estimated pneumonia probability: {prob_neumonia:.2f}%")


if __name__ == "__main__":
    main()

