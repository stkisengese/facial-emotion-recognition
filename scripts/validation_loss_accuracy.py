"""
Plot learning curves for final trained model to check that training 
stopped before clear overfitting (gap between train and val curves 
not too wide at the end).
Shows training stopped before clear overfitting
"""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = 'results'
MODEL_DIR = os.path.join(RESULTS_DIR, 'model')
HISTORY_PATH = os.path.join(MODEL_DIR, 'final_model_history.pkl')
CURVE_PATH = os.path.join(MODEL_DIR, 'learning_curves.png')

def plot_learning_curves():
    if not os.path.exists(HISTORY_PATH):
        print(f"History file not found: {HISTORY_PATH}")
        print("Run train.py first to generate it.")
        return

    with open(HISTORY_PATH, 'rb') as f:
        history = pickle.load(f)

    epochs = range(1, len(history['loss']) + 1)
    
    # Find approximate early stopping point (lowest val_loss)
    best_epoch = np.argmin(history['val_loss']) + 1
    print(f"Best epoch (lowest val loss): {best_epoch}")

    plt.figure(figsize=(14, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Early Stop / Best (epoch {best_epoch})')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Early Stop / Best (epoch {best_epoch})')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Learning curves saved → {CURVE_PATH}")
    print("→ Check that curves show training stopped BEFORE clear overfitting (gap not too wide at the end)")

if __name__ == "__main__":
    plot_learning_curves()
    print("Done.")