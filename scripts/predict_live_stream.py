"""
Real-time facial emotion prediction from webcam or video file.
Predicts at least once per second, prints emotion + probability.
"""

import os
import time
import cv2
import numpy as np
import tensorflow as tf
from preprocess import (
    load_face_cascade,          # ensure it's called or loaded
    detect_and_crop_face,
    EMOTION_LABELS_LIST         # ['Angry', 'Disgust', ..., 'Neutral']
)

# ────────────────────────────────────────────────
#               Configuration
# ────────────────────────────────────────────────
MODEL_PATH = 'results/model/final_emotion_model.keras'  # ← update to your best model, e.g. vgg_style_model.keras

# Load model once
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

EMOTIONS = EMOTION_LABELS_LIST  # list of 7 strings


def draw_prediction(frame, emotion, confidence, bbox=None):
    """Overlay prediction text and bounding box on frame."""

    color = (0, 255, 0) if confidence > 70 else (0, 165, 255)  # green if confident, orange otherwise
    
    text = f"{emotion}: {confidence:.0f}%"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame

def main(video_source=0):
    """
    Main loop for real-time prediction.
    
    video_source: 0 = default webcam, or path to .mp4 file
    """
    print("\n\033[32mReading video stream ...\033[0m")
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open source {video_source}")
        return
    
    # Timing control: predict once per second
    last_prediction_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of video stream or error reading frame.")
            break
        
        current_time = time.time()
        
        # Predict only once per second
        if current_time - last_prediction_time >= 1.0:
            print("\033[32mPreprocessing ...\033[0m")
            
            face, bbox = detect_and_crop_face(frame)
            
            if face is not None:
                # Add batch dimension → (1, 48, 48, 1)
                face_batch = np.expand_dims(face, axis=0)
                
                # Inference
                predictions = model.predict(face_batch, verbose=0)[0]
                emotion_idx = np.argmax(predictions)
                confidence = predictions[emotion_idx] * 100
                
                emotion = EMOTIONS[emotion_idx]
                
                # Print in required format
                timestamp = time.strftime("%H:%M:%S")
                print(f"\n\033[32m{timestamp}s : {emotion} , {confidence:.0f}%\033[0m")
                
                # Overlay on frame
                frame = draw_prediction(frame, emotion, confidence, bbox)
            else:
                print("No face detected")
                # Optional: show message on frame
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            last_prediction_time = current_time
        
        # Always show the live feed
        cv2.imshow('Emotion Detection - Press q to quit', frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames. Session ended.\n")

if __name__ == "__main__":
    import sys
    
    source = 0  # default: webcam
    
    if len(sys.argv) > 1:
        # Use provided video file instead
        source = sys.argv[1]
        print(f"Using video file as input: {source}")
    
    # Make sure cascade is loaded
    load_face_cascade()
    
    main(source)

# #  Usage
# # Webcam (default)
# python scripts/predict_live_stream.py

# # Use a recorded video instead (fallback / test)
# python scripts/predict_live_stream.py path/to/your/test_video_20s.mp4
# python scripts/predict_live_stream.py results/preprocessing_test/input_video.mp4
# Note: If using a video file, ensure it has faces and is not too long (20s max recommended) for testing.