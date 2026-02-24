"""
Focal Loss for handling class imbalance in BUY/HOLD/SELL classification.

Focal Loss down-weights easy examples (HOLD) and focuses learning
on hard, misclassified examples (BUY/SELL minority classes).

Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
"""
import tensorflow as tf


class SparseFocalLoss(tf.keras.losses.Loss):
    """
    Sparse Categorical Focal Loss.
    
    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
               0 = equivalent to cross-entropy
               2 = recommended default for moderate imbalance
        alpha: Class weight balancing (None for no weighting, or list of per-class weights)
    """
    def __init__(self, gamma=2.0, alpha=None, name='sparse_focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        
        # Get true class probabilities
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(tf.squeeze(y_true), depth=tf.shape(y_pred)[-1])
        
        # Cross entropy component
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        
        # Focal weight: (1 - p_t)^gamma
        p_t = tf.reduce_sum(y_pred * y_true_one_hot, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        
        # Apply focal weight
        loss = focal_weight * cross_entropy
        
        # Apply alpha (class weights) if provided
        if self.alpha is not None:
            alpha_tensor = tf.constant(self.alpha, dtype=tf.float32)
            alpha_weight = tf.reduce_sum(alpha_tensor * y_true_one_hot, axis=-1, keepdims=True)
            loss = alpha_weight * loss
        
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha
        })
        return config


if __name__ == "__main__":
    import numpy as np
    
    # Test: HOLD-dominated samples
    y_true = tf.constant([0, 0, 0, 0, 0, 1, 2, 0, 0, 1])  # 7 HOLD, 2 BUY, 1 SELL
    y_pred = tf.constant([
        [0.8, 0.1, 0.1],  # HOLD correct, high conf
        [0.7, 0.2, 0.1],  # HOLD correct, okay conf
        [0.9, 0.05, 0.05],  # HOLD correct, high conf
        [0.6, 0.3, 0.1],  # HOLD correct, low conf
        [0.85, 0.1, 0.05],  # HOLD correct, high conf
        [0.4, 0.5, 0.1],  # BUY correct, low conf
        [0.3, 0.1, 0.6],  # SELL correct, low conf
        [0.8, 0.15, 0.05],  # HOLD correct
        [0.75, 0.2, 0.05],  # HOLD correct
        [0.6, 0.3, 0.1],  # BUY wrong (predicted HOLD)
    ], dtype=tf.float32)
    
    # Compare losses
    ce_loss = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)
    focal_loss = SparseFocalLoss(gamma=2.0)(y_true, y_pred)
    focal_weighted = SparseFocalLoss(gamma=2.0, alpha=[0.5, 2.0, 2.0])(y_true, y_pred)
    
    print(f"Cross-Entropy Loss:       {ce_loss:.4f}")
    print(f"Focal Loss (gamma=2):     {focal_loss:.4f}")
    print(f"Focal + Weights:          {focal_weighted:.4f}")
    print("(Focal should focus training on hard BUY/SELL examples)")
