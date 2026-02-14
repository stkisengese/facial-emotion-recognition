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

# ────────────────────────────────────────────────
#               Build Baseline Model
# ────────────────────────────────────────────────
def build_baseline_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Flatten + Dense
        Flatten(),
        Dense(128),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),

        # Output
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ────────────────────────────────────────────────
#               Callbacks
# ────────────────────────────────────────────────
def get_callbacks():
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOGS_DIR, f"baseline_{timestamp}")

    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        BASELINE_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    return [tensorboard, early_stop, checkpoint]