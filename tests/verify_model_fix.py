
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.lstm_cnn_attention import build_model

print("Testing model build (no label smoothing)...")
model = build_model(
    input_shape=(20, 18), 
    num_classes=3, 
    use_focal_loss=False,
    model_size='light'
)

X = np.random.random((32, 20, 18)).astype(np.float32)
y = np.random.randint(0, 3, (32,)).astype(np.float32)

print("Testing model.fit (2 epochs)...")
model.fit(X, y, epochs=2, batch_size=16, verbose=1)
print("ALL CHECKS PASSED!")
