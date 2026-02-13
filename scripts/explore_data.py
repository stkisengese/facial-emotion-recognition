import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # or 'svg', 'pdf', 'ps', 'cairo', etc.
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
DATA_DIR = 'data'
RESULTS_DIR = 'results'
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

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
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_pixels(pixel_string):
    return np.array(pixel_string.split(), dtype='uint8')

def plot_samples(df, num_samples=5):
    for emotion_idx, emotion_name in EMOTION_LABELS.items():
        emotion_df = df[df['emotion'] == emotion_idx]
        if emotion_df.empty:
            continue

        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        fig.suptitle(f'Emotion: {emotion_name}', fontsize=16)

        samples = emotion_df.sample(min(len(emotion_df), num_samples))

        for i, (_, row) in enumerate(samples.iterrows()):
            pixels = parse_pixels(row['pixels'])
            img = pixels.reshape(48, 48)
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')

        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, f'emotion_{emotion_name}_samples.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")
