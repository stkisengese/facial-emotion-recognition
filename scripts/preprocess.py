"""
Data preprocessing utilities for Facial Emotion Recognition (FER2013 variant).
Handles pixel string parsing, normalization, augmentation, and dataset loading.
"""

import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = 'data'
IMG_SIZE = 48
NUM_CLASSES = 7
RESULTS_DIR = 'results'

EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}
EMOTION_LABELS_LIST = list(EMOTION_LABELS.values())  # ['Angry', 'Disgust', ...]

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

def evaluate_model(model, X, y_true, emotion_labels=None, print_report=True):
    """
    Evaluate model on test set and print key metrics.
    
    Args:
        model: trained Keras model
        X: preprocessed test images (N,48,48,1)
        y_true: one-hot or integer labels
        emotion_labels: list of str names (optional)
        print_report: whether to print detailed classification report
    
    Returns:
        test_accuracy (float)
    """
    if y_true is None:
        print("No ground truth labels → cannot compute accuracy")
        return None
    
    # Predict
    y_pred_prob = model.predict(X, batch_size=64, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true_int = np.argmax(y_true, axis=1) if y_true.ndim == 2 else y_true
    
    acc = accuracy_score(y_true_int, y_pred)
    print(f"\nTest set accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    if print_report and emotion_labels:
        print("\nClassification Report:")
        print(classification_report(y_true_int, y_pred, target_names=emotion_labels))
    
    # confusion matrix plot
    if print_report:
        cm = confusion_matrix(y_true_int, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=emotion_labels, yticklabels=emotion_labels)
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix - Test Set')
        cm_path = os.path.join(RESULTS_DIR, 'model', 'test_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved: {cm_path}")
    
    return acc

# ────────────────────────────────────────────────
#          Face Detection & Preprocessing for Video
# ────────────────────────────────────────────────

# Global cascade classifier (loaded once)
face_cascade = None

def load_face_cascade():
    global face_cascade
    if face_cascade is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(
                f"Haar cascade not found: {cascade_path}\n"
                "OpenCV should include it. If missing, download from:\n"
                "https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml"
            )
        face_cascade = cv2.CascadeClassifier(cascade_path)
        print("Haar Cascade loaded successfully.")
    return face_cascade

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