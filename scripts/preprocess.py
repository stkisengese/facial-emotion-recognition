"""
Data preprocessing utilities for Facial Emotion Recognition (FER2013 variant).
Handles pixel string parsing, normalization, augmentation, and dataset loading.
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
DATA_DIR = 'data'
IMG_SIZE = 48
NUM_CLASSES = 7

EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def parse_pixels(pixel_string):
    """
    Convert space-separated pixel string to numpy array.
    Returns shape (48, 48, 1)
    """
    pixels = np.array(pixel_string.split(), dtype='uint8')
    if len(pixels) != IMG_SIZE * IMG_SIZE:
        raise ValueError(f"Expected {IMG_SIZE*IMG_SIZE} pixels, got {len(pixels)}")
    return pixels.reshape(IMG_SIZE, IMG_SIZE, 1)


def normalize_image(image):
    """Scale pixel values from [0,255] → [0,1] float32."""
    return image.astype('float32') / 255.0

