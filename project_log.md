Set up

```bash
mamba create -n emotion_detector python=3.11 -c conda-forge \
    tensorflow-cpu \
    keras \
    opencv \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    seaborn \
    tensorboard \
    pillow \
    h5py \
    tqdm \
    scipy \
    jupyterlab
```

Verification script
```python
import tensorflow as tf
import keras
import cv2

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}") # Should be 3.x
print(f"OpenCV version: {cv2.__version__}")
print(f"CPU cores detected: {tf.config.list_physical_devices('CPU')}")

# Test if Keras 3 is using the correct backend
print(f"Keras backend: {keras.backend.backend()}")
```