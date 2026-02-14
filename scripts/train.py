"""
Baseline CNN training for Facial Emotion Recognition.
Trains a simple model, saves architecture, and creates initial checkpoint.
"""

import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D,
    Dropout, Flatten, Dense
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    TensorBoard, EarlyStopping, ModelCheckpoint
)
from preprocess import (
    load_and_preprocess_data, DATA_DIR, ensure_dir
)

# ────────────────────────────────────────────────
#               Configuration
# ────────────────────────────────────────────────
RESULTS_DIR = 'results'
MODEL_DIR = os.path.join(RESULTS_DIR, 'model')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')
BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, 'baseline_model.keras')
ARCH_TXT_PATH = os.path.join(MODEL_DIR, 'baseline_arch.txt')

BATCH_SIZE = 64
EPOCHS = 30          # We'll use early stopping so likely < 30
LEARNING_RATE = 0.001

ensure_dir(MODEL_DIR)
ensure_dir(LOGS_DIR)