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

def analyze_distribution(df, set_name="Train"):
    counts = df['emotion'].value_counts().sort_index()
    counts.index = [EMOTION_LABELS[i] for i in counts.index]

    print(f"\n{set_name} Distribution:")
    print(counts)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values, palette='viridis', hue=counts.index, legend=False)
    plt.title(f'{set_name} Class Distribution')
    plt.ylabel('Count')
    plt.xlabel('Emotion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{set_name.lower()}_distribution.png'))
    plt.close()

def main():
    print("Starting Data Exploration...")
    ensure_dir(RESULTS_DIR) # Ensure results directory exists

    if os.path.exists(TRAIN_FILE):
        print(f"Loading {TRAIN_FILE}...")
        train_df = pd.read_csv(TRAIN_FILE)

        print(f"Training samples: {len(train_df)}")
        print(f"Columns: {train_df.columns.tolist()}")

        # Check image shape
        first_img = parse_pixels(train_df.iloc[0]['pixels'])
        print(f"Image shape: {int(np.sqrt(len(first_img)))}x{int(np.sqrt(len(first_img)))}")

        analyze_distribution(train_df, "Train")
        plot_samples(train_df)
    else:
        print(f"Warning: {TRAIN_FILE} not found.")

if __name__ == "__main__":
    main()