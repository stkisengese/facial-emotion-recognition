# GitHub Issues for Face Emotion Classification Project

**Project:** Real-Time Facial Emotion Detection System  
**Technologies:** TensorFlow, Keras, OpenCV, Python  
**Dataset:** Kaggle FER2013 (7 emotion classes)  
**Target Accuracy:** >60% on test set

---

## Issue #1: Project Setup and Environment Configuration

**Labels:** `setup`, `priority-high`, `beginner-friendly`

### Description
Set up the project structure, development environment, and dependencies before starting development.

### Tasks
- [x] Create project directory structure as specified:
  ```
  project/
  ├── data/
  ├── results/
  │   ├── model/
  │   └── preprocessing_test/
  ├── scripts/
  ├── requirements.txt
  └── README.md
  ```
- [x] Set up Python virtual environment (Python 3.8+)
- [x] Create `requirements.txt` with initial dependencies:
  ```
  tensorflow>=2.10.0
  keras>=2.10.0
  opencv-python>=4.6.0
  numpy>=1.23.0
  pandas>=1.5.0
  matplotlib>=3.6.0
  scikit-learn>=1.1.0
  tensorboard>=2.10.0
  seaborn>=0.12.0
  ```
- [x] Install all dependencies: `pip install -r requirements.txt`
- [x] Verify TensorFlow installation: `python -c "import tensorflow as tf; print(tf.__version__)"`
- [x] Verify OpenCV installation: `python -c "import cv2; print(cv2.__version__)"`
- [x] Create `.gitignore` file (ignore data files, model checkpoints, `__pycache__`, etc.)
- [x] Initialize Git repository and create initial commit

### Acceptance Criteria
- Virtual environment activated
- All dependencies installed without errors
- Project structure matches requirements
- Can import TensorFlow, Keras, and OpenCV successfully

### Estimated Time
30 minutes

---

## Issue #2: Download and Explore Emotion Dataset

**Labels:** `data`, `priority-high`, `exploration`  
**Depends on:** #1

### Description
Download the Kaggle emotion detection dataset and perform exploratory data analysis (EDA).

### Tasks
- [x] Download dataset from [Kaggle FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- [x] Place `train.csv` and `test.csv` in `data/` folder
- [x] Create exploratory notebook/script: `scripts/explore_data.py`
- [x] Load and inspect data structure:
  ```python
  import pandas as pd
  train_df = pd.read_csv('data/train.csv')
  print(train_df.head())
  print(train_df.info())
  print(train_df['emotion'].value_counts())
  ```
- [x] Verify data format:
  - Check image dimensions (should be 48×48 pixels)
  - Check grayscale (single channel)
  - Check pixel value ranges (0-255)
- [x] Analyze class distribution (check for imbalance)
- [x] Visualize sample images from each emotion class
- [x] Create visualization: Plot 5 random samples per emotion class
- [x] Document findings in `README.md`:
  - Total training samples
  - Total test samples
  - Class distribution
  - Any data quality issues

### Sample Code
```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize samples
def plot_emotion_samples(df, emotion_label, num_samples=5):
    emotion_data = df[df['emotion'] == emotion_label]
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        pixels = np.array(emotion_data.iloc[i]['pixels'].split(), dtype='uint8')
        image = pixels.reshape(48, 48)
        axes[i].imshow(image, cmap='gray')
        axes[i].axis('off')
    plt.savefig(f'results/emotion_{emotion_label}_samples.png')
```

### Acceptance Criteria
- Dataset downloaded and placed in correct folder
- Data structure understood (pixels column, emotion labels)
- Class distribution documented
- Sample visualizations created
- Any data issues identified and documented

### Estimated Time
1 hour

---

## Issue #3: Create Data Preprocessing Pipeline

**Labels:** `data`, `priority-high`, `preprocessing`  
**Depends on:** #2

### Description
Build a robust data preprocessing pipeline to prepare images for CNN training.

### Tasks
- [x] Create `scripts/preprocess.py` with preprocessing functions
- [x] Implement function to parse pixel strings to numpy arrays:
  ```python
  def parse_pixels(pixel_string):
      pixels = np.array(pixel_string.split(), dtype='uint8')
      return pixels.reshape(48, 48, 1)  # Add channel dimension
  ```
- [x] Implement normalization (scale 0-255 → 0-1):
  ```python
  def normalize_image(image):
      return image.astype('float32') / 255.0
  ```
- [x] Create train/validation split from train.csv (80/20 or 90/10)
- [x] Implement data augmentation using `ImageDataGenerator`:
  - Rotation range: ±10 degrees
  - Width/height shift: 10%
  - Horizontal flip: True
  - Zoom range: 10%
- [x] Create function to load and preprocess entire dataset:
  ```python
  def load_and_preprocess_data(csv_path, augment=False):
      # Load CSV
      # Parse pixels
      # Normalize
      # Convert labels to categorical
      # Return X, y
  ```
- [x] Convert emotion labels to one-hot encoding (7 classes)
- [x] Verify preprocessed data shape: `(num_samples, 48, 48, 1)`
- [x] Save preprocessing functions for reuse in prediction scripts
- [x] Test preprocessing pipeline with small sample

### Acceptance Criteria
- Preprocessing functions work correctly
- Data augmentation implemented
- Train/validation split created
- Images normalized to [0, 1] range
- Labels in one-hot encoded format
- Pipeline tested and validated

### Estimated Time
2 hours

---

## Issue #4: Build Baseline CNN Architecture

**Labels:** `model`, `priority-high`, `cnn`  
**Depends on:** #3

### Description
Implement a baseline CNN architecture to establish performance benchmark before optimization.

### Tasks
- [x] Create `scripts/train.py` for model training
- [x] Implement simple baseline CNN (similar to MNIST experience):
  ```python
  model = Sequential([
      Conv2D(32, (3,3), padding="same", input_shape=(48,48,1)),
      BatchNormalization(),
      LeakyReLU(alpha=0.1),
      MaxPooling2D(2,2),
      Dropout(0.25),
      
      Conv2D(64, (3,3), padding="same"),
      BatchNormalization(),
      LeakyReLU(alpha=0.1),
      MaxPooling2D(2,2),
      Dropout(0.25),
      
      Flatten(),
      Dense(128),
      BatchNormalization(),
      LeakyReLU(alpha=0.1),
      Dropout(0.5),
      Dense(7, activation='softmax')
  ])
  ```
- [x] Compile model:
  - Optimizer: Adam (lr=0.001)
  - Loss: categorical_crossentropy
  - Metrics: accuracy
- [x] Print `model.summary()` and save to `results/model/baseline_arch.txt`
- [x] Train for 10-20 epochs (quick baseline)
- [x] Evaluate on test set
- [x] Document baseline accuracy in `README.md`
- [x] Save model as `baseline_model.keras`

### Acceptance Criteria
- Baseline model trains without errors
- Model summary saved
- Test accuracy documented (target: >45%)
- Model saved successfully
- Training completes in reasonable time (<30 min)

### Estimated Time
1.5 hours

---

## Issue #5: Integrate TensorBoard Monitoring (MANDATORY)

**Labels:** `monitoring`, `priority-critical`, `tensorboard`  
**Depends on:** #4

### Description
Set up TensorBoard for real-time training monitoring. This is **MANDATORY** per project requirements.

### Tasks
- [x] Create TensorBoard log directory: `results/logs/`
- [x] Implement TensorBoard callback in `train.py`:
  ```python
  from tensorflow.keras.callbacks import TensorBoard
  from datetime import datetime
  
  log_dir = f"results/logs/fit_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
  tensorboard_callback = TensorBoard(
      log_dir=log_dir,
      histogram_freq=1,
      write_graph=True,
      write_images=True,
      update_freq='epoch'
  )
  ```
- [x] Add callback to model.fit():
  ```python
  history = model.fit(
      ...,
      callbacks=[tensorboard_callback, ...]
  )
  ```
- [x] Test TensorBoard launch: `tensorboard --logdir results/logs`
- [x] Verify you can view:
  - Loss curves (train and validation)
  - Accuracy curves (train and validation)
  - Model graph
  - Histograms of weights/biases
- [x] Document TensorBoard access in `README.md`
- [ ] Take screenshot during training and save as `results/model/tensorboard.png` (**REQUIRED**)

### Additional Monitoring
- [x] Add custom metrics logging (optional):
  - Per-class accuracy
  - Confusion matrix callback

### Acceptance Criteria
- TensorBoard callback integrated
- Can launch TensorBoard and view metrics
- Screenshot saved showing training progress
- All metrics display correctly
- Documentation updated

### Estimated Time
45 minutes

---

## Issue #6: Implement Early Stopping and Callbacks

**Labels:** `training`, `priority-high`, `callbacks`  
**Depends on:** #5

### Description
Prevent overfitting by implementing early stopping and model checkpointing.

### Tasks
- [x] Implement Early Stopping callback:
  ```python
  from tensorflow.keras.callbacks import EarlyStopping
  
  early_stop = EarlyStopping(
      monitor='val_loss',
      patience=10,
      restore_best_weights=True,
      verbose=1
  )
  ```
- [x] Implement ModelCheckpoint callback:
  ```python
  from tensorflow.keras.callbacks import ModelCheckpoint
  
  checkpoint = ModelCheckpoint(
      'results/model/best_model.keras',
      monitor='val_accuracy',
      save_best_only=True,
      mode='max',
      verbose=1
  )
  ```
- [x] Implement ReduceLROnPlateau (optional but recommended):
  ```python
  from tensorflow.keras.callbacks import ReduceLROnPlateau
  
  reduce_lr = ReduceLROnPlateau(
      monitor='val_loss',
      factor=0.5,
      patience=5,
      min_lr=1e-7,
      verbose=1
  )
  ```
- [x] Combine all callbacks:
  ```python
  callbacks = [
      tensorboard_callback,
      early_stop,
      checkpoint,
      reduce_lr
  ]
  ```
- [x] Test training with callbacks (run for many epochs, verify early stopping triggers)
- [x] Document callback configuration in `README.md`

### Acceptance Criteria
- Early stopping prevents overfitting
- Best model weights saved automatically
- Learning rate reduces when plateauing
- Training stops at optimal point
- Callbacks documented

### Estimated Time
1 hour

---

## Issue #7: Create Learning Curves Visualization

**Labels:** `visualization`, `priority-high`, `plotting`  
**Depends on:** #6

### Description
Create and save learning curves showing training stopped before overfitting (REQUIRED deliverable).

### Tasks
- [x] Create `scripts/validation_loss_accuracy.py`
- [x] Extract training history after model.fit():
  ```python
  history = model.fit(...)
  ```
- [x] Plot training vs validation loss:
  ```python
  import matplotlib.pyplot as plt
  
  plt.figure(figsize=(12, 4))
  
  # Plot 1: Loss
  plt.subplot(1, 2, 1)
  plt.plot(history.history['loss'], label='Training Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  plt.axvline(x=best_epoch, color='r', linestyle='--', label='Early Stop')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.title('Training and Validation Loss')
  
  # Plot 2: Accuracy
  plt.subplot(1, 2, 2)
  plt.plot(history.history['accuracy'], label='Training Accuracy')
  plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
  plt.axvline(x=best_epoch, color='r', linestyle='--', label='Early Stop')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.title('Training and Validation Accuracy')
  
  plt.tight_layout()
  plt.savefig('results/model/learning_curves.png', dpi=300)
  ```
- [x] Verify plot shows:
  - Clear divergence point (where overfitting would start)
  - Early stopping marker
  - Final achieved accuracy
- [x] Save as `results/model/learning_curves.png` (**REQUIRED**)
- [x] Add plot to `README.md`

### Acceptance Criteria
- Learning curves plotted correctly
- Shows both loss and accuracy
- Early stopping point visible
- High-quality image saved (300 DPI)
- Demonstrates model stopped before overfitting

### Estimated Time
45 minutes

---

## Issue #8: Optimize CNN Architecture (Achieve >60% Accuracy)

**Labels:** `model`, `priority-critical`, `optimization`  
**Depends on:** #7

### Description
Iteratively improve CNN architecture to achieve >60% test accuracy (REQUIRED).

### Tasks
- [ ] Document current best accuracy as baseline
- [ ] **Iteration 1:** Increase model depth
  - Add third convolutional block (128 filters)
  - Test and document results
- [ ] **Iteration 2:** Adjust regularization
  - Experiment with dropout rates (0.2-0.5)
  - Try SpatialDropout2D in conv layers
  - Test and document results
- [ ] **Iteration 3:** Try different architectures
  - VGG-style (more conv layers, smaller filters)
  - ResNet-style (add skip connections - requires Functional API)
  - Test and document results
- [ ] **Iteration 4:** Optimize hyperparameters
  - Learning rate: try 0.0001, 0.001, 0.01
  - Batch size: try 32, 64, 128
  - Optimizer: try Adam, RMSprop, SGD with momentum
  - Test and document results
- [ ] **Iteration 5:** Enhanced data augmentation
  - Increase rotation range
  - Add random brightness adjustment
  - Test and document results
- [ ] Track ALL experiments in a table:
  | Architecture | Params | Train Acc | Val Acc | Test Acc | Notes |
  |--------------|--------|-----------|---------|----------|-------|
  | Baseline     | 50K    | 55%       | 48%     | 47%      | Underfitting |
  | VGG-style    | 200K   | 68%       | 58%     | 57%      | Some overfitting |
  | ...          | ...    | ...       | ...     | ...      | ... |
- [ ] Select best performing architecture (>60% test accuracy)
- [ ] Save final model as `results/model/final_emotion_model.keras`
- [ ] Document architecture iterations in `results/model/final_emotion_model_arch.txt`

### Architecture Example (if struggling)
```python
model = Sequential([
    # Block 1
    Conv2D(64, (3,3), padding="same", input_shape=(48,48,1)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Conv2D(64, (3,3), padding="same"),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(2,2),
    SpatialDropout2D(0.2),
    
    # Block 2
    Conv2D(128, (3,3), padding="same"),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Conv2D(128, (3,3), padding="same"),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(2,2),
    SpatialDropout2D(0.3),
    
    # Block 3
    Conv2D(256, (3,3), padding="same"),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Conv2D(256, (3,3), padding="same"),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(2,2),
    SpatialDropout2D(0.4),
    
    # Classifier
    GlobalAveragePooling2D(),
    Dense(256),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.5),
    Dense(7, activation='softmax')
])
```

### Acceptance Criteria
- Test accuracy > 60% (MANDATORY)
- All iterations documented
- Best model saved
- Architecture explanation written
- Learning curves show proper training

### Estimated Time
4-6 hours (including training time)

---

## Issue #9: Create Model Architecture Documentation

**Labels:** `documentation`, `priority-high`, `deliverable`  
**Depends on:** #8

### Description
Document final CNN architecture with explanations (REQUIRED deliverable).

### Tasks
- [ ] Create `results/model/final_emotion_model_arch.txt`
- [ ] Include `model.summary()` output
- [ ] Write detailed explanation covering:
  - **Architecture Overview:** Brief description
  - **Input Layer:** Shape and preprocessing
  - **Convolutional Blocks:** 
    - Number of blocks
    - Filters per block
    - Why this progression?
  - **Activation Functions:** LeakyReLU vs ReLU choice
  - **Regularization:** 
    - BatchNormalization placement
    - Dropout rates and reasoning
    - Data augmentation used
  - **Pooling Strategy:** MaxPooling vs GlobalAveragePooling
  - **Classifier Head:** Dense layer sizes
  - **Output Layer:** Softmax for 7 classes
- [ ] Document iteration process:
  - What didn't work and why
  - What did work and why
  - How you arrived at final architecture
- [ ] Include hyperparameters:
  - Learning rate
  - Batch size
  - Optimizer choice
  - Loss function
- [ ] Add performance metrics:
  - Final training accuracy
  - Final validation accuracy
  - Final test accuracy
  - Training time
  - Number of parameters

### Template
```
=== FINAL EMOTION CLASSIFICATION MODEL ARCHITECTURE ===

Test Accuracy: 62.5%
Training Time: 45 minutes (30 epochs)
Total Parameters: 1,234,567

=== MODEL SUMMARY ===
[Paste model.summary() output here]

=== ARCHITECTURE DECISIONS ===

1. Input Processing:
   - 48x48 grayscale images normalized to [0,1]
   - Data augmentation: rotation, shift, zoom

2. Convolutional Blocks:
   - 3 blocks with progressive filter increase (64→128→256)
   - Each block: Conv→BN→LeakyReLU→Conv→BN→LeakyReLU→Pool→Dropout
   - Reasoning: Deeper networks learn hierarchical features...

3. Regularization:
   - SpatialDropout2D in conv layers (0.2, 0.3, 0.4)
   - Regular Dropout in dense layer (0.5)
   - BatchNormalization after each convolution
   - Reasoning: Emotion data limited, prevent overfitting...

[Continue for all aspects...]

=== ITERATION HISTORY ===
Baseline (3 layers): 47% → underfitting
Added depth (4 layers): 55% → better but still underfit
Increased filters: 61% → close!
Added data augmentation: 62.5% → SUCCESS!
```

### Acceptance Criteria
- Complete architecture documentation
- Clear explanations for design choices
- Iteration history included
- Performance metrics documented
- Professional formatting

### Estimated Time
1.5 hours

---

## Issue #10: Implement Face Detection for Video Preprocessing

**Labels:** `opencv`, `priority-high`, `preprocessing`  
**Depends on:** #8

### Description
Implement face detection using OpenCV to preprocess video frames (required for real-time prediction).

### Tasks
- [x] Research face detection methods:
  - Haar Cascade (fastest, good for frontal faces)
  - dlib (more accurate but slower)
  - MTCNN (most accurate but slowest)
- [x] Choose Haar Cascade for speed (recommended)
- [x] Download Haar Cascade XML file:
  ```python
  import cv2
  # OpenCV includes this by default
  face_cascade = cv2.CascadeClassifier(
      cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
  )
  ```
- [x] Implement face detection function in `scripts/preprocess.py`:
  ```python
  def detect_face(frame):
      """
      Detect face in frame and return cropped face image
      
      Args:
          frame: BGR image from video
      Returns:
          face_image: 48x48 grayscale image or None
      """
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(
          gray,
          scaleFactor=1.1,
          minNeighbors=5,
          minSize=(48, 48)
      )
      
      if len(faces) == 0:
          return None
      
      # Take largest face
      (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
      
      # Crop face
      face = gray[y:y+h, x:x+w]
      
      # Resize to 48x48
      face_resized = cv2.resize(face, (48, 48))
      
      # Normalize
      face_normalized = face_resized.astype('float32') / 255.0
      
      # Add channel dimension
      face_final = np.expand_dims(face_normalized, axis=-1)
      
      return face_final
  ```
- [x] Test face detection on sample images:
  - Test with frontal face
  - Test with side face
  - Test with multiple faces
  - Test with no face
- [x] Handle edge cases:
  - No face detected → skip frame or use previous detection
  - Multiple faces → use largest face
  - Face too small → skip frame
- [x] Visualize detection (draw rectangle around detected face)
- [x] Document face detection parameters in code

### Acceptance Criteria
- Face detection works reliably on various images
- Outputs 48×48 grayscale normalized images
- Handles edge cases gracefully
- Code documented with comments
- Tested on multiple scenarios

### Estimated Time
2 hours

---

## Issue #11: Create Preprocessing Test Pipeline

**Labels:** `testing`, `priority-high`, `video-processing`  
**Depends on:** #10

### Description
Build preprocessing test that extracts 1 face image per second from a 20-second video (REQUIRED deliverable).

### Tasks
- [x] Record or download a 20-second test video with a face
- [ ] Save as `results/preprocessing_test/input_video.mp4`
- [x] Implement video processing in `scripts/preprocess.py`:
  ```python
  def preprocess_video(video_path, output_dir, fps=1):
      """
      Extract and preprocess faces from video
      
      Args:
          video_path: Path to input video
          output_dir: Directory to save images
          fps: Frames per second to extract (1 = 1 image/sec)
      """
      cap = cv2.VideoCapture(video_path)
      video_fps = cap.get(cv2.CAP_PROP_FPS)
      frame_interval = int(video_fps / fps)
      
      frame_count = 0
      saved_count = 0
      
      while True:
          ret, frame = cap.read()
          if not ret:
              break
          
          # Process every Nth frame
          if frame_count % frame_interval == 0:
              face_image = detect_face(frame)
              
              if face_image is not None:
                  # Save as grayscale PNG
                  save_path = f"{output_dir}/image{saved_count}.png"
                  cv2.imwrite(save_path, face_image * 255)  # Denormalize for saving
                  saved_count += 1
                  print(f"Saved: {save_path}")
          
          frame_count += 1
      
      cap.release()
      print(f"Total frames processed: {frame_count}")
      print(f"Total faces saved: {saved_count}")
  ```
- [x] Run preprocessing test:
  ```bash
  python scripts/preprocess.py
  ```
- [x] Verify output:
  - 20-21 images in `results/preprocessing_test/`
  - Each image is 48×48 grayscale
  - Faces properly centered and cropped
  - File naming: `image0.png`, `image1.png`, ..., `image20.png`
- [x] Visualize all extracted faces in a grid (optional but recommended)
- [x] Document preprocessing test in `README.md`

### Acceptance Criteria
- 20-21 face images extracted from video
- All images are 48×48 grayscale
- Images saved in correct directory
- Preprocessing completes without errors
- Output matches expected format

### Estimated Time
1.5 hours

---

## 🔮 Issue #12: Implement Batch Prediction Script

**Labels:** `prediction`, `priority-high`, `testing`  
**Depends on:** #11

### Description
Create script to test model accuracy on test set (REQUIRED: `predict.py`).

### Tasks
- [x] Create `scripts/predict.py`
- [x] Load trained model:
  ```python
  from tensorflow import keras
  
  model = keras.models.load_model('results/model/final_emotion_model.keras')
  ```
- [x] Load and preprocess test data:
  ```python
  from preprocess import load_and_preprocess_data
  
  X_test, y_test = load_and_preprocess_data('data/test.csv')
  ```
- [x] Make predictions:
  ```python
  predictions = model.predict(X_test)
  predicted_classes = np.argmax(predictions, axis=1)
  true_classes = np.argmax(y_test, axis=1)
  ```
- [x] Calculate accuracy:
  ```python
  from sklearn.metrics import accuracy_score
  
  accuracy = accuracy_score(true_classes, predicted_classes)
  print(f"Accuracy on test set: {accuracy*100:.0f}%")
  ```
- [x] Add confusion matrix (optional but recommended):
  ```python
  from sklearn.metrics import confusion_matrix, classification_report
  
  cm = confusion_matrix(true_classes, predicted_classes)
  print("\nConfusion Matrix:")
  print(cm)
  
  emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
  print("\nClassification Report:")
  print(classification_report(true_classes, predicted_classes, target_names=emotion_labels))
  ```
- [x] Expected output format:
  ```
  Accuracy on test set: 62%
  ```
- [x] Test script: `python scripts/predict.py`
- [x] Verify output matches expected format

### Acceptance Criteria
- Script loads model successfully
- Calculates test accuracy
- Outputs in required format
- Accuracy > 60%
- Script runs without errors

### Estimated Time
1 hour

---

## 🎬 Issue #13: Implement Real-Time Webcam Prediction

**Labels:** `opencv`, `priority-critical`, `real-time`  
**Depends on:** #12

### Description
Create real-time emotion prediction from webcam stream (REQUIRED: `predict_live_stream.py`).

### Tasks
- [x] Create `scripts/predict_live_stream.py`
- [x] Load trained model:
  ```python
  model = keras.models.load_model('results/model/final_emotion_model.keras')
  emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
  ```
- [x] Initialize webcam:
  ```python
  cap = cv2.VideoCapture(0)  # 0 for default webcam
  
  if not cap.isOpened():
      print("Error: Could not open webcam")
      exit()
  ```
- [x] Implement main prediction loop:
  ```python
  import time
  
  print("Reading video stream ...")
  
  frame_count = 0
  fps = 1  # Predict once per second
  prev_time = time.time()
  
  while True:
      ret, frame = cap.read()
      if not ret:
          break
      
      current_time = time.time()
      
      # Predict once per second
      if current_time - prev_time >= 1.0:
          print("Preprocessing ...")
          
          # Detect and preprocess face
          face_image = detect_face(frame)
          
          if face_image is not None:
              # Add batch dimension
              face_batch = np.expand_dims(face_image, axis=0)
              
              # Predict
              predictions = model.predict(face_batch, verbose=0)
              emotion_idx = np.argmax(predictions[0])
              confidence = predictions[0][emotion_idx] * 100
              
              # Print result
              timestamp = time.strftime("%H:%M:%S")
              print(f"{timestamp} : {emotion_labels[emotion_idx]} , {confidence:.0f}%")
          else:
              print("No face detected")
          
          prev_time = current_time
      
      # Display video (optional)
      cv2.imshow('Emotion Detection', frame)
      
      # Exit on 'q' key
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  
  cap.release()
  cv2.destroyAllWindows()
  ```
- [x] Add visual feedback (draw emotion on frame):
  ```python
  # After prediction, add text to frame
  cv2.putText(frame, f"{emotion_labels[emotion_idx]}: {confidence:.0f}%", 
              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  ```
- [x] Handle fallback to video file if webcam unavailable:
  ```python
  import sys
  
  if len(sys.argv) > 1:
      cap = cv2.VideoCapture(sys.argv[1])  # Use video file
  else:
      cap = cv2.VideoCapture(0)  # Use webcam
  ```
- [x] Test with webcam
- [x] Test with recorded video file
- [x] Verify output format matches requirements:
  ```
  Reading video stream ...
  Preprocessing ...
  11:11:11s : Happy , 73%
  Preprocessing ...
  11:11:12s : Happy , 93%
  ...
  ```

### Acceptance Criteria
- Reads from webcam successfully
- Predicts at least 1 emotion per second
- Output format matches requirements
- Works with video file as fallback
- Visual display of predictions
- Graceful error handling

### Estimated Time
2 hours

---

## 📸 Issue #14: Add Visual Enhancements to Live Stream

**Labels:** `opencv`, `enhancement`, `priority-medium`  
**Depends on:** #13

### Description
Enhance real-time prediction with visual overlays and better UX.

### Tasks
- [ ] Draw bounding box around detected face:
  ```python
  def detect_and_draw_face(frame):
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
      
      for (x, y, w, h) in faces:
          cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
      
      return frame, faces
  ```
- [ ] Display emotion label and confidence on frame
- [ ] Add color-coded boxes (e.g., green for Happy, red for Angry)
- [ ] Show prediction probabilities bar chart on frame (optional):
  ```python
  # Create probability bar chart overlay
  y_offset = 50
  for i, (emotion, prob) in enumerate(zip(emotion_labels, predictions[0])):
      bar_length = int(prob * 200)
      cv2.rectangle(frame, (10, y_offset), (10 + bar_length, y_offset + 20), 
                   (255, 0, 0), -1)
      cv2.putText(frame, f"{emotion}: {prob*100:.0f}%", 
                 (220, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
      y_offset += 25
  ```
- [ ] Add FPS counter
- [ ] Add instructions on frame (Press 'q' to quit)
- [ ] Smooth predictions (average last 3 predictions to reduce jitter)
- [ ] Test all visual enhancements

### Acceptance Criteria
- Face bounding box visible
- Emotion label clearly displayed
- Visual feedback responsive
- No significant performance degradation
- Professional-looking UI

### Estimated Time
2 hours

---

## 📊 Issue #15: Generate Confusion Matrix and Analysis

**Labels:** `analysis`, `visualization`, `priority-medium`  
**Depends on:** #12

### Description
Create detailed performance analysis with confusion matrix and per-class metrics.

### Tasks
- [ ] Create `scripts/analyze_results.py`
- [ ] Generate confusion matrix heatmap:
  ```python
  import seaborn as sns
  import matplotlib.pyplot as plt
  from sklearn.metrics import confusion_matrix
  
  cm = confusion_matrix(true_classes, predicted_classes)
  
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
              xticklabels=emotion_labels,
              yticklabels=emotion_labels)
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.title('Emotion Classification Confusion Matrix')
  plt.savefig('results/model/confusion_matrix.png', dpi=300)
  ```
- [ ] Calculate per-class metrics:
  ```python
  from sklearn.metrics import precision_recall_fscore_support
  
  precision, recall, f1, support = precision_recall_fscore_support(
      true_classes, predicted_classes
  )
  
  # Create DataFrame for easy viewing
  import pandas as pd
  metrics_df = pd.DataFrame({
      'Emotion': emotion_labels,
      'Precision': precision,
      'Recall': recall,
      'F1-Score': f1,
      'Support': support
  })
  print(metrics_df)
  metrics_df.to_csv('results/model/per_class_metrics.csv', index=False)
  ```
- [ ] Identify which emotions are confused most often
- [ ] Visualize per-class accuracy bar chart
- [ ] Analyze failure cases:
  - Which emotions are hardest to classify?
  - What are common misclassifications?
- [ ] Document findings in `README.md`

### Acceptance Criteria
- Confusion matrix saved as image
- Per-class metrics calculated
- Analysis documented
- Insights about model performance clear

### Estimated Time
1.5 hours

---

## 📚 Issue #16: Complete Project Documentation

**Labels:** `documentation`, `priority-high`, `deliverable`  
**Depends on:** #15

### Description
Write comprehensive README.md with project overview, setup instructions, and usage guide.

### Tasks
- [x] Update `README.md` with complete documentation
- [x] Include sections:
  - **Project Title and Description**
  - **Emotion Classes:** List of 7 emotions
  - **Dataset:** Link to Kaggle, statistics
  - **Model Architecture:** Brief overview, link to detailed doc
  - **Performance:** Final test accuracy, confusion matrix
  - **Project Structure:** Directory tree
  - **Setup Instructions:**
    ```bash
    # Clone repository
    git clone <repo-url>
    cd face-emotion-classification
    
    # Create virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    
    # Install dependencies
    pip install -r requirements.txt
    
    # Download dataset to data/ folder
    ```
  - **Training:**
    ```bash
    python scripts/train.py
    
    # Monitor with TensorBoard
    tensorboard --logdir results/logs
    ```
  - **Testing:**
    ```bash
    # Test accuracy
    python scripts/predict.py
    
    # Real-time prediction
    python scripts/predict_live_stream.py
    
    # Use video file instead of webcam
    python scripts/predict_live_stream.py path/to/video.mp4
    ```
  - **Results:** Include images (learning curves, confusion matrix)
  - **Architecture Decisions:** Link to detailed architecture doc
  - **Future Improvements:** Ideas for enhancement
  - **References:** Citations, helpful resources
  - **License:** Choose appropriate license
  - **Author:** Your information
- [ ] Add badges (optional):
  - Python version
  - TensorFlow version
  - License
  - Test accuracy
- [ ] Include screenshots:
  - Real-time prediction in action
  - TensorBoard dashboard
  - Learning curves
- [ ] Proofread for clarity and completeness

### Acceptance Criteria
- README is comprehensive and clear
- All sections included
- Setup instructions tested
- Screenshots included
- Professional formatting

### Estimated Time
2 hours

---

## 🧪 Issue #17: Test Complete Pipeline End-to-End

**Labels:** `testing`, `priority-high`, `qa`  
**Depends on:** #16

### Description
Perform comprehensive testing of entire project to ensure all requirements met.

### Checklist
- [ ] **Directory Structure:**
  - [ ] All required folders exist
  - [ ] All required files present
- [ ] **Model Training:**
  - [ ] `final_emotion_model.keras` exists
  - [ ] Model loads without errors
  - [ ] Test accuracy > 60%
- [ ] **Documentation:**
  - [ ] `final_emotion_model_arch.txt` complete
  - [ ] Architecture iterations explained
  - [ ] `learning_curves.png` saved
  - [ ] Shows training stopped before overfitting
  - [ ] `tensorboard.png` saved
  - [ ] Shows TensorBoard in use
- [ ] **Preprocessing Test:**
  - [ ] `results/preprocessing_test/input_video.mp4` exists
  - [ ] 20-21 face images extracted
  - [ ] All images are 48×48 grayscale
  - [ ] Images properly preprocessed
- [ ] **Prediction Scripts:**
  - [ ] `predict.py` works:
    ```bash
    python scripts/predict.py
    # Output: Accuracy on test set: 62%
    ```
  - [ ] `predict_live_stream.py` works:
    ```bash
    python scripts/predict_live_stream.py
    # Output: Timestamps with emotions
    ```
  - [ ] Real-time prediction at 1 FPS
  - [ ] Output format matches requirements
- [ ] **Code Quality:**
  - [ ] All scripts have proper imports
  - [ ] No hardcoded paths (use relative paths)
  - [ ] Error handling implemented
  - [ ] Code commented appropriately
- [ ] **README:**
  - [ ] Setup instructions work
  - [ ] Usage examples accurate
  - [ ] All sections complete
- [ ] **Clean Repository:**
  - [ ] No unnecessary files
  - [ ] `.gitignore` configured properly
  - [ ] Git history clean

### Test Commands
```bash
# Full test sequence
python scripts/train.py          # Should train successfully
python scripts/predict.py        # Should show >60% accuracy
python scripts/predict_live_stream.py  # Should predict from webcam

# Verify TensorBoard
tensorboard --logdir results/logs

# Check file structure
ls -R results/
```

### Acceptance Criteria
- All required files present
- All scripts run without errors
- Test accuracy > 60%
- Documentation complete
- Output formats correct
- Repository clean and organized

### Estimated Time
2 hours

---

## 🌟 Issue #18 (OPTIONAL): Transfer Learning with Pre-trained Model

**Labels:** `enhancement`, `priority-low`, `transfer-learning`, `optional`  
**Depends on:** #8

### Description
Use transfer learning to potentially improve accuracy (OPTIONAL - does not replace building from scratch).

### Tasks
- [ ] Research suitable pre-trained models:
  - VGG16 (trained on ImageNet)
  - ResNet50 (trained on ImageNet)
  - MobileNetV2 (lighter, faster)
  - EfficientNet (state-of-the-art)
- [ ] Load pre-trained model without top layers:
  ```python
  from tensorflow.keras.applications import VGG16
  
  base_model = VGG16(
      weights='imagenet',
      include_top=False,
      input_shape=(48, 48, 3)  # Need RGB for pre-trained models
  )
  
  # Freeze base layers
  base_model.trainable = False
  ```
- [ ] Convert grayscale to RGB (repeat channel 3 times):
  ```python
  def grayscale_to_rgb(image):
      return np.repeat(image, 3, axis=-1)
  ```
- [ ] Add custom classification head:
  ```python
  from tensorflow.keras import Model
  
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(256, activation='relu')(x)
  x = Dropout(0.5)(x)
  outputs = Dense(7, activation='softmax')(x)
  
  model = Model(inputs=base_model.input, outputs=outputs)
  ```
- [ ] Train with frozen base:
  - Compile and train (10-20 epochs)
  - Evaluate performance
- [ ] Fine-tune (unfreeze top layers):
  ```python
  # Unfreeze last few layers
  base_model.trainable = True
  for layer in base_model.layers[:-4]:
      layer.trainable = False
  
  # Re-compile with lower learning rate
  model.compile(
      optimizer=Adam(learning_rate=1e-5),
      loss='categorical_crossentropy',
      metrics=['accuracy']
  )
  
  # Train again
  ```
- [ ] Compare with from-scratch model
- [ ] Document architecture in `pre_trained_model_architecture.txt`
- [ ] Save model as `pre_trained_model.pkl`
- [ ] Add comparison to README

### Acceptance Criteria
- Pre-trained model implemented
- Performance compared to from-scratch
- Architecture documented
- Model saved

### Estimated Time
3-4 hours

---

## 🎯 Issue #19 (OPTIONAL): CNN Adversarial Attack

**Labels:** `enhancement`, `priority-low`, `research`, `optional`, `advanced`  
**Depends on:** #17

### Description
Hack the CNN: Make it predict wrong emotion with slight image modifications (OPTIONAL - very cool!).

### Tasks
- [ ] Research adversarial attacks:
  - FGSM (Fast Gradient Sign Method)
  - PGD (Projected Gradient Descent)
  - C&W attack
- [ ] Implement FGSM attack:
  ```python
  import tensorflow as tf
  
  def create_adversarial_pattern(model, image, true_label, target_label):
      """
      Create adversarial perturbation to change prediction
      
      Args:
          model: Trained emotion model
          image: Original image (predicted as true_label)
          true_label: Current prediction
          target_label: Desired prediction (e.g., Sad instead of Happy)
      """
      image_tensor = tf.convert_to_tensor(image)
      target_one_hot = tf.one_hot(target_label, 7)
      
      with tf.GradientTape() as tape:
          tape.watch(image_tensor)
          prediction = model(image_tensor)
          loss = tf.keras.losses.categorical_crossentropy(target_one_hot, prediction)
      
      # Get gradient
      gradient = tape.gradient(loss, image_tensor)
      
      # Create perturbation
      signed_grad = tf.sign(gradient)
      
      return signed_grad
  
  def generate_adversarial_image(model, image, true_label, target_label, epsilon=0.01):
      """
      Generate adversarial example
      
      Args:
          epsilon: Perturbation magnitude (try 0.01, 0.05, 0.1)
      """
      perturbation = create_adversarial_pattern(model, image, true_label, target_label)
      
      # Add perturbation
      adversarial_image = image + epsilon * perturbation
      
      # Clip to valid range
      adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
      
      return adversarial_image
  ```
- [ ] Find a Happy prediction
- [ ] Generate adversarial example to make it Sad
- [ ] Visualize original vs adversarial:
  ```python
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
  
  # Original
  axes[0].imshow(original_image.squeeze(), cmap='gray')
  axes[0].set_title(f'Original: {original_prediction}')
  
  # Adversarial
  axes[1].imshow(adversarial_image.squeeze(), cmap='gray')
  axes[1].set_title(f'Adversarial: {adversarial_prediction}')
  
  # Difference (amplified)
  diff = np.abs(adversarial_image - original_image) * 10
  axes[2].imshow(diff.squeeze(), cmap='hot')
  axes[2].set_title('Perturbation (amplified 10x)')
  
  plt.savefig('results/adversarial_attack.png')
  ```
- [ ] Document attack in README
- [ ] Discuss implications for model robustness

### Acceptance Criteria
- Successfully flip prediction with minimal changes
- Original and adversarial images look nearly identical to humans
- Perturbation visualized
- Attack documented

### Estimated Time
3-4 hours

---

## 🚀 Issue #20: Final Submission Preparation

**Labels:** `deliverable`, `priority-critical`, `final`  
**Depends on:** #17

### Description
Final checklist before submission to ensure all requirements met.

### Pre-Submission Checklist

#### **Required Files:**
- [ ] `data/train.csv` (downloaded, not committed)
- [ ] `data/test.csv` (downloaded, not committed)
- [ ] `results/model/final_emotion_model.keras`
- [ ] `results/model/final_emotion_model_arch.txt`
- [ ] `results/model/learning_curves.png`
- [ ] `results/model/tensorboard.png` ⚠️ **MANDATORY**
- [ ] `results/preprocessing_test/input_video.mp4`
- [ ] `results/preprocessing_test/image0.png` through `image20.png`
- [ ] `scripts/train.py`
- [ ] `scripts/predict.py`
- [ ] `scripts/predict_live_stream.py`
- [ ] `scripts/preprocess.py`
- [ ] `scripts/validation_loss_accuracy.py`
- [ ] `requirements.txt`
- [ ] `README.md`

#### **Performance Requirements:**
- [ ] Test accuracy > 60% ✅
- [ ] TensorBoard integrated ✅
- [ ] Early stopping implemented ✅
- [ ] Learning curves show stopped before overfitting ✅

#### **Script Functionality:**
- [ ] `python scripts/predict.py` outputs: `Accuracy on test set: XX%`
- [ ] `python scripts/predict_live_stream.py` works with webcam
- [ ] Real-time prediction at 1 FPS minimum
- [ ] Output format matches requirements

#### **Code Quality:**
- [ ] No syntax errors
- [ ] Proper imports
- [ ] Error handling
- [ ] Comments and documentation
- [ ] Consistent code style

#### **Documentation:**
- [ ] README complete and accurate
- [ ] Architecture explained clearly
- [ ] Setup instructions tested
- [ ] Usage examples work

#### **Git Repository:**
- [ ] `.gitignore` configured (exclude large files, data, models if >100MB)
- [ ] Clean commit history
- [ ] Descriptive commit messages
- [ ] No sensitive information

### Final Test Commands
```bash
# Clean install test
git clone <your-repo>
cd face-emotion-classification
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run all scripts
python scripts/train.py
python scripts/predict.py
python scripts/predict_live_stream.py

# Verify outputs
ls -la results/model/
ls -la results/preprocessing_test/
```

### Acceptance Criteria
- All required files present
- All scripts run successfully
- Performance requirements met
- Documentation complete
- Repository clean and professional

### Estimated Time
1-2 hours

---

# 📅 Recommended Timeline

**Week 1: Setup & Data (Issues #1-3)**
- Day 1: Setup environment, download data
- Day 2: Data exploration, preprocessing pipeline

**Week 2: Model Development (Issues #4-9)**
- Day 3-4: Baseline model, TensorBoard, callbacks
- Day 5-7: Architecture optimization (achieve >60%)
- Day 8: Documentation

**Week 3: Video Processing (Issues #10-14)**
- Day 9-10: Face detection, preprocessing test
- Day 11-12: Prediction scripts, real-time stream
- Day 13: Visual enhancements

**Week 4: Analysis & Polish (Issues #15-20)**
- Day 14: Confusion matrix, analysis
- Day 15: Complete documentation
- Day 16: End-to-end testing
- Day 17-18: Optional features (transfer learning, adversarial)
- Day 19-20: Final review and submission

**Total Estimated Time:** 40-50 hours

---

# 🎯 Success Metrics

- ✅ Test accuracy > 60%
- ✅ All required files present
- ✅ Scripts run without errors
- ✅ TensorBoard integrated
- ✅ Real-time prediction works
- ✅ Documentation complete
- ✅ Professional code quality

---

# 💡 Tips for Success

1. **Start Early:** Don't underestimate training time
2. **Monitor Training:** TensorBoard is your friend
3. **Document Everything:** Keep notes on what works/doesn't work
4. **Test Frequently:** Run scripts after each major change
5. **Version Control:** Commit often with meaningful messages
6. **Ask for Help:** If stuck >2 hours, seek assistance
7. **Stay Organized:** Follow the project structure strictly
8. **Backup Models:** Save checkpoints during long training runs

---

# 📚 Helpful Resources

- [Keras Documentation](https://keras.io/)
- [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
- [TensorFlow TensorBoard Guide](https://www.tensorflow.org/tensorboard)
- [Kaggle FER2013 Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
- [CNN Architectures Guide](https://keras.io/examples/vision/)

---
