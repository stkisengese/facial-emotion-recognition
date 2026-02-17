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
```bash
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
    Try transfer learning (MobileNetV2, EfficientNet)
    Add class weights to handle imbalance
    Use MTCNN or MediaPipe for better face detection
    Implement temporal smoothing (average last 3–5 frames)
    Deploy as web app (Streamlit / Flask) or Android app
    Explore video-level emotion (RNN/LSTM on frame sequences)

## License
[MIT License](LICENSE) – feel free to use, modify, and learn from this project.
✉️ Contact / Connect

## Contact/Connect
GitHub: github.com/SKisengese/facial-emotion-recognition
X / Twitter: @SKisengese
Location: Mombasa, Kenya

