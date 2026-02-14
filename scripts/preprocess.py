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


def load_and_preprocess_data(
    csv_path,
    split='train',
    val_split=0.2,
    augment=False,
    seed=42
):
    """
    Load CSV, preprocess images/labels, apply optional augmentation.
    
    Args:
        csv_path: Path to train.csv or test_with_emotions.csv
        split: 'train' or 'test'
        val_split: Fraction for validation split (only used if split='train')
        augment: Whether to return ImageDataGenerator (for training only)
        seed: Random seed for reproducibility
    
    Returns:
        If split == 'train':
            (X_train, y_train), (X_val, y_val), datagen (or None)
        If split == 'test':
            X_test, y_test (or None if no labels)
    """
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if 'pixels' not in df.columns:
        raise ValueError("CSV missing 'pixels' column")
    
    # Parse all images
    print("Parsing pixels...")
    X = np.array([parse_pixels(p) for p in df['pixels']])
    X = normalize_image(X)  # → (N, 48, 48, 1)
    
    # Labels
    if 'emotion' in df.columns:
        y = df['emotion'].values
        y = to_categorical(y, num_classes=NUM_CLASSES)
        has_labels = True
    else:
        y = None
        has_labels = False
        print("No emotion labels found → test set without ground truth")
    
    if split == 'test':
        return X, y if has_labels else None
    
    # For training: split into train/val
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_split,
        random_state=seed,
        stratify=np.argmax(y, axis=1) if y is not None else None
    )
    
    datagen = None
    if augment:
        print("Creating data augmentation generator...")
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        datagen.fit(X_train)
    
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    return (X_train, y_train), (X_val, y_val), datagen


def get_test_data(labeled=True):
    """
    Convenience function to load test set.
    Uses test_with_emotions.csv if labeled=True, else test.csv
    """
    if labeled:
        path = os.path.join(DATA_DIR, 'test_with_emotions.csv')
    else:
        path = os.path.join(DATA_DIR, 'test.csv')
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test file not found: {path}")
    
    X, y = load_and_preprocess_data(path, split='test')
    return X, y


# Test the preprocessing pipeline
if __name__ == "__main__":
    ensure_dir('results/preprocessing_test')
    
    print("Testing preprocessing pipeline...")
    
    try:
        (X_tr, y_tr), (X_v, y_v), _ = load_and_preprocess_data(
            os.path.join(DATA_DIR, 'train.csv'),
            split='train',
            augment=True
        )
        
        # Save one sample image to verify
        import matplotlib.pyplot as plt
        plt.imsave(
            'results/preprocessing_test/sample_normalized.png',
            X_tr[0].squeeze(),
            cmap='gray'
        )
        print("Sample normalized image saved → results/preprocessing_test/sample_normalized.png")
        
        # Quick stats
        print(f"Train samples: {len(X_tr)}, Val samples: {len(X_v)}")
        
    except Exception as e:
        print(f"Preprocessing test failed: {e}")