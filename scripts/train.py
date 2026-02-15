"""
Baseline CNN training for Facial Emotion Recognition.
Trains a simple model, saves architecture, and creates initial checkpoint.
"""

import os
import datetime
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D,
    Dropout, Flatten, Dense
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from preprocess import (
    load_and_preprocess_data, DATA_DIR, RESULTS_DIR,
    ensure_dir, evaluate_model, EMOTION_LABELS_LIST
)

# ────────────────────────────────────────────────
#               Configuration
# ────────────────────────────────────────────────
MODEL_DIR = os.path.join(RESULTS_DIR, 'model')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')
BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, 'baseline_model.keras')
ARCH_TXT_PATH = os.path.join(MODEL_DIR, 'baseline_arch.txt')

BATCH_SIZE = 64
EPOCHS = 50          # We'll use early stopping so likely < 30
LEARNING_RATE = 0.001

ensure_dir(MODEL_DIR)
ensure_dir(LOGS_DIR)

# ────────────────────────────────────────────────
#               Build Baseline Model
# ────────────────────────────────────────────────
def build_baseline_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential([
        # Block 1
        Input(shape=input_shape),
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        # Flatten + Dense
        Flatten(),
        Dense(128),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),

        # Output
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ────────────────────────────────────────────────
#               Callbacks
# ────────────────────────────────────────────────
def get_callbacks():
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOGS_DIR, f"baseline_{timestamp}")

    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1,
        mode='min',
        min_delta=0.001       # Only stop if improvement is less than 0.1% to avoid stopping on minor fluctuations
    )

    checkpoint = ModelCheckpoint(
        BASELINE_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5, # Reduce learning rate by half if validation loss doesn't improve for 4 epochs
        patience=4,
        min_lr=1e-5, # Don't reduce below this learning rate
        verbose=1,
        cooldown=2   # Wait for 2 epochs after reducing LR before monitoring again to avoid rapid reductions
    )

    return [tensorboard, early_stop, checkpoint, reduce_lr]

# ────────────────────────────────────────────────
#               Main Training Flow
# ────────────────────────────────────────────────
def main():
    print("Starting Baseline CNN Training...")

    # Load data with augmentation
    (X_train, y_train), (X_val, y_val), datagen = load_and_preprocess_data(
        os.path.join(DATA_DIR, 'train.csv'),
        split='train',
        augment=True,
        val_split=0.2
    )

    model = build_baseline_model()
    model.summary()

    # Save architecture to text file
    with open(ARCH_TXT_PATH, 'w') as f:
        f.write("=== BASELINE MODEL ARCHITECTURE (Issue #4) ===\n\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\n\nHyperparameters:\n")
        f.write(f"Batch size:     {BATCH_SIZE}\n")
        f.write(f"Learning rate:  {LEARNING_RATE}\n")
        f.write(f"Max epochs:     {EPOCHS}\n")
        f.write(f"Optimizer:      Adam\n")
        f.write(f"Augmentation:   Yes (rotation±10°, shift/zoom 10%, hflip)\n")

    print(f"Model architecture saved to: {ARCH_TXT_PATH}")

    callbacks = get_callbacks()

    print("Starting training...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # ────────────────────────────────────────────────
    #  Quick test-set evaluation after training
    # ────────────────────────────────────────────────
    print("\nEvaluating on labeled test set (test_with_emotions.csv)...")

    X_test, y_test = load_and_preprocess_data(
        os.path.join(DATA_DIR, 'test_with_emotions.csv'),
        split='test'
    )

    if y_test is not None:
        test_acc = evaluate_model(model, X_test, y_test, EMOTION_LABELS_LIST)
        print(f"→ This is the number we'll try to push >60% in iterations: {test_acc:.4f}")

    # Save training history for later analysis (validation curves)
    history_dict = history.history
    with open(os.path.join(MODEL_DIR, 'baseline_history.pkl'), 'wb') as f:
        pickle.dump(history_dict, f)
    print(f"Training history saved to: {os.path.join(MODEL_DIR, 'baseline_history.pkl')}")

    # Quick final evaluation on validation set
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"Model saved (best weights) at: {BASELINE_MODEL_PATH}")

    print("\nDone. Next steps:")
    print("1. Check results/model/baseline_arch.txt")
    print("2. Launch TensorBoard: tensorboard --logdir results/logs")
    print("3. Take screenshot → save as results/model/tensorboard.png")


if __name__ == "__main__":
    tf.random.set_seed(42)   # reproducibility
    np.random.seed(42)
    main()