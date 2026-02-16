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


def draw_prediction(frame, emotion, confidence, bbox=None):
    """Overlay prediction text and bounding box on frame."""
    
    color = (0, 255, 0) if confidence > 70 else (0, 165, 255)  # green if confident, orange otherwise
    
    text = f"{emotion}: {confidence:.0f}%"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame