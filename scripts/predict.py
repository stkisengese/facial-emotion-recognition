"""
Evaluate model on test set and print accuracy in the required format.
"""

import os
import sys
from preprocess import (
    get_test_data,
    evaluate_model,
    EMOTION_LABELS_LIST,
    DATA_DIR
)

# ────────────────────────────────────────────────
#               Configuration
# ────────────────────────────────────────────────
MODEL_PATH = 'results/model/final_emotion_model.keras' 
TEST_FILE_LABELED = os.path.join(DATA_DIR, 'test_with_emotions.csv')


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please update MODEL_PATH to your trained model file.")
        sys.exit(1)

    if not os.path.exists(TEST_FILE_LABELED):
        print(f"Error: Labeled test file not found: {TEST_FILE_LABELED}")
        sys.exit(1)

    print("Loading model...")
    from tensorflow import keras
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded.")

    print("\nLoading labeled test set...")
    X_test, y_test = get_test_data(labeled=True)

    print("\nEvaluating on test set...")
    acc = evaluate_model(
        model,
        X_test,
        y_test,
        emotion_labels=EMOTION_LABELS_LIST,
        print_report=True   # shows classification report + saves confusion matrix
    )

    if acc is not None:
        # Exactly the required output format
        print(f"\nAccuracy on test set: {acc*100:.0f}%")


if __name__ == "__main__":
    main()