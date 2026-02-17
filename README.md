# Real-Time Facial Emotion Recognition

![Python](https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg?style=flat-square)
![Keras](https://img.shields.io/badge/Keras-2.10+-red.svg?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.6+-green.svg?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)
![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-65.4%25-brightgreen.svg?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-success.svg?style=flat-square)

**End-to-end deep learning project: Detect 7 facial emotions in real time from webcam or video using a custom CNN trained on FER2013.**

![Demo Screenshot](results/model/live_demo_screenshot.png)  
*(Live webcam prediction – Happy 92% with bounding box overlay)*

## Project Overview

This project builds a **real-time facial emotion recognition system** capable of classifying seven emotions:  
**Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**

It demonstrates a full machine learning pipeline:
- Data exploration & preprocessing
- Custom CNN architecture from scratch (no transfer learning required)
- Training with regularization, early stopping, learning rate reduction & TensorBoard monitoring
- Face detection & real-time inference using OpenCV
- Evaluation reaching **>60% test accuracy** (final: **65.4%** on labeled test set)

Built as both an academic challenge solution and a **strong portfolio piece** showcasing computer vision, deep learning engineering, and deployment skills.

## Key Features

- Real-time emotion prediction from webcam (~1 FPS minimum)
- Face detection & centering using OpenCV Haar Cascade
- Data augmentation & stratified train/val split
- Early stopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
- Clean modular code structure (`scripts/`, `results/`, `data/`)
- Comprehensive documentation & reproducibility

## Results Highlights

**Final Model Performance** (test_with_emotions.csv – 7,178 samples)

| Metric              | Value     |
|---------------------|-----------|
| Test Accuracy       | 65.4%     |
| Macro Avg F1        | 0.61      |
| Happy Precision     | 0.84      |
| Happy Recall        | 0.89      |
| Disgust F1          | 0.48      |
| Fear F1             | 0.40      |

**Confusion Matrix**  
![Confusion Matrix](results/model/final_test_confusion_matrix.png)

**Learning Curves** (showing early stopping before overfitting)  
![Learning Curves](results/model/learning_curves.png)

**TensorBoard Dashboard**  
![TensorBoard](results/model/tensorboard.png)

**Live Demo Example**  
![Live Prediction](results/model/live_demo_screenshot.png)

## Project Structure
facial-emotion-recognition/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── test_with_emotions.csv
├── results/
│   ├── model/
│   │   ├── final_emotion_model.keras
│   │   ├── final_emotion_model_arch.txt
│   │   ├── learning_curves.png
│   │   ├── tensorboard.png
│   │   └── final_test_confusion_matrix.png
│   └── preprocessing_test/
│       ├── input_video.mp4
│       ├── image0.png ... imageN.png
├── scripts/
│   ├── train.py
│   ├── predict.py
│   ├── predict_live_stream.py
│   ├── preprocess.py
│   └── validation_loss_accuracy.py
├── requirements.txt
└── README.md


## Quick Start

### Prerequisites

- Python 3.11+
- Webcam (or fallback video file)
- ~4–8 GB RAM (GPU strongly recommended for training)

### Installation

**Option 1: Conda/Mamba (recommended)**

```bash
# Create & activate environment
mamba env create -f environment.yml    # if you have one
# or manually:
mamba create -n fer-emotion python=3.11
mamba activate fer-emotion
mamba install tensorflow opencv pandas numpy matplotlib seaborn scikit-learn

# Install remaining pip packages
pip install -r requirements.txt
```

**Option 2: venv + Pip**
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

**Option 3: Google Colab/Kaggle Notebooks**
Open a new notebook → upload repo → run:
```bash
!pip install -r requirements.txt
```

### Download Dataset
```bash
wget https://assets.01-edu.org/ai-branch/project3/emotions-detector.zip
unzip emotions-detector.zip -d data/
```
Place train.csv, test.csv, test_with_emotions in data/.

## Usage

### 1. Train the model
```bash
python scripts/train.py

# Monitor progress via TensorBoard
tensorboard --logdir results/model/logs
```

### 2. Evaluate on the test set
```bash
python scripts/predict.py
```
Expected output
```bash
Accuracy on the test set: 65%
```

### 3. Run real-time webcam demo
```bash
python scripts/predict_live_stream.py
```
- Press q to quit
- Fallback to video file:
```python
python scripts/predict_live_stream.py path/to/video.mp4

# example from saved video in project folder
python scripts/predict_live_stream.py results/preprocessing_test/input_video.mp4
```

## Model Architecture & Decisions
**Final Model:** VGG-style stacked convolutional blocks (32 → 64 → 128 filters), LeakyReLU activations, BatchNorm, Dropout (0.25–0.5), GlobalAveragePooling or Flatten + Dense(256) head.

**Why this architecture?**
- Baseline (2 blocks) → ~55–56% test
- Added depth + more filters → +8–10% lift
- LeakyReLU + SpatialDropout → better gradient flow on small/imbalanced data
- Early stopping + LR reduction → prevented overfitting

Iteration Summary (see results/model/final_emotion_model_arch.txt for full details)

## Model Architecture & Selection Process
<details>
<summary>Click to expand final architecture report</summary>

```md
FINAL EMOTION CLASSIFICATION MODEL ARCHITECTURE
================================================

Project: Real-Time Facial Emotion Recognition (FER2013 Dataset)
Goal Achieved: Test accuracy > 60% (final: 65.6%)
Model File: results/model/final_emotion_model.keras

1. ARCHITECTURE OVERVIEW
------------------------
Input: 48×48 grayscale images (1 channel), normalized to [0,1]
Backbone: 3 progressive convolutional blocks (VGG-style)
  - Filters: 32 → 64 → 128
  - Per block: Conv2D → BatchNorm → LeakyReLU(0.1) → Conv2D → BatchNorm → LeakyReLU(0.1) → MaxPooling2D → Dropout/SpatialDropout2D
Pooling: MaxPooling2D after each block
Flatten → Dense(256) → BatchNorm → LeakyReLU → Dropout(0.5)
Output: Dense(7, softmax) for 7-class classification

2. KEY DESIGN DECISIONS
-----------------------
- LeakyReLU(α=0.1) used instead of ReLU → prevents dying neurons, better gradient flow on small/imbalanced FER2013 data
- BatchNormalization after every Conv → stabilizes activations and speeds convergence
- SpatialDropout2D(0.25–0.3) in deeper blocks → drops entire feature maps → stronger regularization against noisy labels
- Dropout(0.25–0.3) in conv blocks + Dropout(0.5) in dense layer → prevents overfitting
- Flatten + Dense(256) chosen over GlobalAveragePooling2D → GAP reduced accuracy (~63–64%); larger dense layer improved separation of ambiguous classes (Sad ↔ Neutral, Fear ↔ Surprise)
- MaxPooling instead of strided conv → preserves spatial hierarchy while reducing dimensions efficiently

3. HYPERPARAMETERS & TRAINING SETUP
-----------------------------------
Optimizer: Adam (initial lr=0.001)
Loss: Categorical Crossentropy
Batch size: 64
Callbacks:
  - EarlyStopping(patience=8, min_delta=0.001, restore_best_weights=True)
  - ModelCheckpoint(monitor='val_accuracy', save_best_only=True)
  - ReduceLROnPlateau(factor=0.5, patience=4, cooldown=2, min_lr=1e-6)
Training time: ~70 minutes on Colab T4 GPU
Total parameters: ~1.4 million

4. ITERATION HISTORY & MODEL SELECTION
--------------------------------------
Experiment 000 – Baseline CNN
  - 2 conv blocks (32→64), Dense(128)
  - Test acc: ~57–58%
  - Issue: underfitting, mild overfitting after ~18 epochs
  - Decision: too shallow → increase depth

Experiment 001 – Deeper CNN
  - Added 3rd block (128 filters)
  - Test acc: ~62%
  - Decision: clear improvement → continue deepening

Experiment 002 – VGG-style stacking
  - Each block: 2 conv layers
  - Test acc: ~64%
  - Decision: richer features → new baseline

Experiment 003 – VGG + SpatialDropout2D + Dense(256)
  - Added SpatialDropout2D(0.25–0.3), increased dense to 256
  - Test acc: 65.5–65.7% (most consistent runs)
  - Decision: best balance of capacity, regularization, and stability → SELECTED AS FINAL MODEL

Experiment 004 – 4-block variant (added 256-filter block)
  - Test acc: ~66.4%
  - Issue: slight gain but unstable validation curves
  - Decision: dropped (diminishing returns, risk of overfitting)

Experiments 005–009 – Ablations (GAP, multi-dense, dropout tuning)
  - Test acc: ~63–64%
  - Decision: confirmed Flatten + single Dense(256) outperforms alternatives

Final selection rationale:
  Experiment 003 consistently delivered 65–66% test accuracy with stable training curves and early stopping before overfitting.
  It provides sufficient depth for hierarchical features, strong regularization for noisy/imbalanced data, and adequate classifier capacity for ambiguous emotions.

5. FINAL PERFORMANCE METRICS
----------------------------
Training Accuracy (final epoch): ~68%
Validation Accuracy: ~66%
Test Accuracy (test_with_emotions.csv): 65.6%
Training Time: ~70 minutes (Colab T4 GPU)
Parameters: ~1.4 million

This architecture represents the optimal trade-off between model capacity, generalization, and training stability for the FER2013 emotion detection task under the given constraints.

```
</details>

## Performance & Limitations
**Strengths**
- Reaches project goal (>60%) without transfer learning
- Real-time capable on modest hardware
- Robust face centering & normalization

**Limitations**
- Imbalanced dataset → Disgust & Fear underperform
- Haar Cascade → struggles with profile/low-light faces
- No temporal modeling (single-frame prediction)

## Future Improvements
- Try transfer learning (MobileNetV2, EfficientNet)
- Add class weights to handle imbalance
- Use MTCNN or MediaPipe for better face detection
- Implement temporal smoothing (average last 3–5 frames)
- Deploy as web app (Streamlit / Flask) or Android app
- Explore video-level emotion (RNN/LSTM on frame sequences)

## License
[MIT License](LICENSE) – feel free to use, modify, and learn from this project.
✉️ Contact / Connect

## Contact/Connect
    GitHub: github.com/SKisengese/facial-emotion-recognition
    X / Twitter: @SKisengese

