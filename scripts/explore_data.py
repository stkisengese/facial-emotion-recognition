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
