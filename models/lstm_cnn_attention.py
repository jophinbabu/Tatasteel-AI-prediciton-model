"""
Adaptive LSTM-CNN-Attention Model Architecture

Automatically selects model size based on available training data:
  - 'light'  (<1000 samples): ~15K params — prevents overfitting on small data
  - 'medium' (1000-5000):     ~100K params — balanced
  - 'full'   (>5000):         ~490K params — maximum capacity
"""
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Dropout,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D,
    BatchNormalization, Add
)
from tensorflow.keras.regularizers import l2

# Add project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ── Model size configs ──
MODEL_CONFIGS = {
    'light': {
        'conv1_filters': 24,
        'conv2_filters': 48,
        'lstm1_units': 48,
        'lstm2_units': 24,
        'attn_heads': 2,
        'attn_key_dim': 16,
        'dense1_units': 48,
        'dense2_units': 24,
        'dropout': 0.5,       # Higher dropout for small data
        'l2_reg': 1e-3,       # Stronger regularization
    },
    'medium': {
        'conv1_filters': 32,
        'conv2_filters': 64,
        'lstm1_units': 64,
        'lstm2_units': 32,
        'attn_heads': 2,
        'attn_key_dim': 32,
        'dense1_units': 64,
        'dense2_units': 32,
        'dropout': 0.45,
        'l2_reg': 5e-4,
    },
    'full': {
        'conv1_filters': 64,
        'conv2_filters': 128,
        'lstm1_units': 128,
        'lstm2_units': 64,
        'attn_heads': 4,
        'attn_key_dim': 64,
        'dense1_units': 128,
        'dense2_units': 64,
        'dropout': 0.3,
        'l2_reg': 1e-4,
    }
}


def select_model_size(n_samples):
    """Automatically select model size based on training data volume."""
    if n_samples < 4000:
        return 'light'
    elif n_samples < 8000:
        return 'medium'
    else:
        return 'full'


def build_model(input_shape=(30, 40), num_classes=3, use_focal_loss=True,
                focal_gamma=2.0, class_weights_alpha=None, learning_rate=0.001,
                model_size='auto', n_samples=None):
    """
    Builds the LSTM-CNN-Attention model with adaptive sizing.
    
    Args:
        input_shape: (sequence_length, num_features)
        num_classes: Number of output classes (3: HOLD/BUY/SELL)
        use_focal_loss: Use focal loss for class imbalance
        focal_gamma: Focal loss gamma parameter
        class_weights_alpha: Per-class weights [hold, buy, sell]
        learning_rate: Learning rate for AdamW
        model_size: 'light', 'medium', 'full', or 'auto'
        n_samples: Number of training samples (used when model_size='auto')
    """
    # Auto-select model size
    if model_size == 'auto':
        model_size = select_model_size(n_samples or 0)
    
    cfg = MODEL_CONFIGS.get(model_size, MODEL_CONFIGS['medium'])
    print(f"Building {model_size.upper()} model (cfg: conv={cfg['conv1_filters']}/{cfg['conv2_filters']}, "
          f"lstm={cfg['lstm1_units']}/{cfg['lstm2_units']}, "
          f"dense={cfg['dense1_units']}/{cfg['dense2_units']}, drop={cfg['dropout']})")
    
    inputs = Input(shape=input_shape, name='input')
    
    # ── 1. CNN Feature Extraction ──
    x = Conv1D(filters=cfg['conv1_filters'], kernel_size=3, activation='relu', 
               padding='same', name='conv1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = MaxPooling1D(pool_size=2, name='pool1')(x)
    x = Dropout(cfg['dropout'], name='drop_conv1')(x)
    
    x = Conv1D(filters=cfg['conv2_filters'], kernel_size=3, activation='relu', 
               padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Dropout(cfg['dropout'], name='drop_conv2')(x)
    
    # ── 2. LSTM with Residual ──
    lstm_input = x
    x = Bidirectional(LSTM(units=cfg['lstm1_units'], return_sequences=True, 
                           name='bilstm'), name='bidirectional')(x)
    x = BatchNormalization(name='bn_lstm1')(x)
    x = LSTM(units=cfg['lstm2_units'], return_sequences=True, name='lstm2')(x)
    
    residual = Conv1D(filters=cfg['lstm2_units'], kernel_size=1, padding='same', 
                      name='residual_proj')(lstm_input)
    x = Add(name='residual_add')([x, residual])
    x = LayerNormalization(name='ln_residual')(x)
    
    # ── 3. Self-Attention ──
    attn1 = MultiHeadAttention(num_heads=cfg['attn_heads'], key_dim=cfg['attn_key_dim'], 
                                name='mha1')(x, x)
    x = LayerNormalization(name='ln_attn1')(attn1 + x)
    x = Dropout(0.2, name='drop_attn1')(x)
    
    # Second attention only for medium/full
    if model_size in ('medium', 'full'):
        attn2 = MultiHeadAttention(num_heads=max(2, cfg['attn_heads'] // 2), 
                                    key_dim=max(16, cfg['attn_key_dim'] // 2), 
                                    name='mha2')(x, x)
        x = LayerNormalization(name='ln_attn2')(attn2 + x)
        x = Dropout(0.2, name='drop_attn2')(x)
    
    # ── 4. Classification Head ──
    x = GlobalAveragePooling1D(name='gap')(x)
    
    x = Dense(cfg['dense1_units'], activation='relu', 
              kernel_regularizer=l2(cfg['l2_reg']), name='dense1')(x)
    x = BatchNormalization(name='bn_dense1')(x)
    x = Dropout(cfg['dropout'], name='drop_dense1')(x)
    
    x = Dense(cfg['dense2_units'], activation='relu', 
              kernel_regularizer=l2(cfg['l2_reg']), name='dense2')(x)
    x = Dropout(cfg['dropout'], name='drop_dense2')(x)
    
    outputs = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=f'LSTM_CNN_Attn_{model_size}')
    
    # ── 5. Compile ──
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=cfg['l2_reg'])
    
    if use_focal_loss:
        from models.focal_loss import SparseFocalLoss
        loss = SparseFocalLoss(gamma=focal_gamma, alpha=class_weights_alpha)
    else:
        loss = 'sparse_categorical_crossentropy'
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    trainable = sum(p.numpy().size for p in model.trainable_weights)
    print(f"  Trainable parameters: {trainable:,}")
    
    return model


if __name__ == "__main__":
    for size in ('light', 'medium', 'full'):
        print(f"\n{'='*50}")
        model = build_model(input_shape=(20, 49), model_size=size)
        print()
