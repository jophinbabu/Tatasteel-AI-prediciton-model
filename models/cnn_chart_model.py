"""
CNN Chart Image Classifier

Custom lightweight CNN for classifying candlestick chart images into
BUY / HOLD / SELL. Designed for ~3K training images.

Architecture:
    Input (224×224×3)
    → Conv2D(32, 3×3) + BN + ReLU + MaxPool
    → Conv2D(64, 3×3) + BN + ReLU + MaxPool
    → Conv2D(128, 3×3) + BN + ReLU + MaxPool
    → Conv2D(64, 3×3) + BN + ReLU + MaxPool
    → GlobalAveragePooling2D
    → Dense(64) + Dropout
    → Dense(3, softmax)
    
~80K parameters — sized for small datasets.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def build_cnn_model(input_shape=(224, 224, 3), num_classes=3, learning_rate=0.001):
    """
    Builds a lightweight CNN for chart image classification.
    """
    inputs = Input(shape=input_shape, name='chart_input')
    
    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = MaxPooling2D((2, 2), name='pool1')(x)
    x = Dropout(0.25, name='drop1')(x)
    
    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = MaxPooling2D((2, 2), name='pool2')(x)
    x = Dropout(0.25, name='drop2')(x)
    
    # Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = MaxPooling2D((2, 2), name='pool3')(x)
    x = Dropout(0.3, name='drop3')(x)
    
    # Block 4
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv4')(x)
    x = BatchNormalization(name='bn4')(x)
    x = MaxPooling2D((2, 2), name='pool4')(x)
    x = Dropout(0.3, name='drop4')(x)
    
    # Classification head
    x = GlobalAveragePooling2D(name='gap')(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4), name='dense1')(x)
    x = Dropout(0.5, name='drop_dense')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='ChartCNN')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    trainable = sum(p.numpy().size for p in model.trainable_weights)
    print(f"ChartCNN: {trainable:,} trainable parameters")
    
    return model


def load_chart_dataset(image_dir, img_size=(224, 224), batch_size=32, 
                       validation_split=0.15, seed=42):
    """
    Loads chart images from labeled directories using tf.keras.
    
    Expected structure:
        image_dir/
            BUY/
            HOLD/
            SELL/
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        image_dir,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        batch_size=batch_size,
        validation_split=validation_split,
        subset='training',
        seed=seed,
        class_names=['HOLD', 'BUY', 'SELL']
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        image_dir,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        batch_size=batch_size,
        validation_split=validation_split,
        subset='validation',
        seed=seed,
        class_names=['HOLD', 'BUY', 'SELL']
    )
    
    # Normalize pixel values to [0, 1]
    normalization = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))
    
    # Prefetch for performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds


def train_cnn(image_dir='data/chart_images', epochs=50, batch_size=32,
              model_save_path='models/saved/chart_cnn.keras'):
    """
    Trains the CNN chart classifier.
    """
    print("=" * 50)
    print("Training Chart Image CNN")
    print("=" * 50)
    
    # Load data
    train_ds, val_ds = load_chart_dataset(image_dir, batch_size=batch_size)
    
    # Count class distribution for class weights
    label_counts = {0: 0, 1: 0, 2: 0}
    for _, labels in train_ds:
        for label in labels.numpy():
            label_counts[int(label)] = label_counts.get(int(label), 0) + 1
    
    total = sum(label_counts.values())
    n_classes = len(label_counts)
    class_weights = {c: total / (n_classes * cnt) for c, cnt in label_counts.items() if cnt > 0}
    print(f"Class weights: {class_weights}")
    
    # Build model
    model = build_cnn_model()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1
        ),
    ]
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Evaluate
    from sklearn.metrics import classification_report
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0).argmax(axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
    
    print("\nCNN Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['HOLD', 'BUY', 'SELL'], zero_division=0))
    
    return model, history


if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()
