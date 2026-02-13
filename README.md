# Emotion Detection CNN

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.10+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.6+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Accuracy](https://img.shields.io/badge/accuracy-62%25-brightgreen.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

> Real-time facial emotion detection system using Convolutional Neural Networks

A deep learning-based facial emotion recognition system that detects and classifies emotions in real-time from webcam feeds. The project implements a Convolutional Neural Network (CNN) trained on the FER2013 dataset to identify seven distinct facial emotions with over 60% accuracy.

Key Features:
- Real-time emotion detection from webcam at 1 FPS
- Custom CNN architecture optimized for emotion classification
- Comprehensive training pipeline with TensorBoard monitoring
- Face detection and preprocessing using OpenCV
- Seven emotion categories: Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral

This project demonstrates end-to-end machine learning workflow including data preprocessing, model training with early stopping, real-time inference, and deployment-ready code.

## 📊 Data Analysis

The model is trained on the FER2013 dataset.

- **Image Properties**: 48x48 pixels, Grayscale (1 channel).
- **Pixel Range**: 0-255.
- **Emotions**: 0:Angry, 1:Disgust, 2:Fear, 3:Happy, 4:Sad, 5:Surprise, 6:Neutral.

### Class Distribution
The dataset is known to be imbalanced, with 'Happy' being the most represented and 'Disgust' the least.

| Emotion | Train Samples |
|---------|---------------|
| Angry   | ~3,995        |
| Disgust | ~436          |
| Fear    | ~4,097        |
| Happy   | ~7,215        |
| Sad     | ~4,830        |
| Surprise| ~3,171        |
| Neutral | ~4,965        |

##  Quick Start

### Option 1: Mamba/Conda (Recommended - Matches my setup)
```bash
mamba env create -f environment.yml
mamba activate cnn_env
```

### Option 2: Pip/Venv (Lightweight)
bash
```python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Option 3: Google Colab (GPU)
```
!pip install -r requirements.txt
```

### Verify Setup
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import keras; print(keras.__version__)"
python -c "import numpy as np; print(np.__version__)"
python -c "import cv2; print(cv2.__version__)"

jupyter lab  # Start notebook
```
