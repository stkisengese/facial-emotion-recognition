"""
Real-time facial emotion prediction from webcam or video file.
Predicts at least once per second, prints emotion + probability.
"""

import os
import time
import cv2
import numpy as np
import tensorflow as tf
from preprocess import (
    load_face_cascade,          # ensure it's called or loaded
    detect_and_crop_face,
    EMOTION_LABELS_LIST         # ['Angry', 'Disgust', ..., 'Neutral']
)

# ────────────────────────────────────────────────
#               Configuration
# ────────────────────────────────────────────────
MODEL_PATH = 'results/model/baseline_model.keras'  # ← update to your best model, e.g. vgg_style_model.keras

# Load model once
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

EMOTIONS = EMOTION_LABELS_LIST  # list of 7 strings