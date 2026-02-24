"""
Ensemble Model Module

Trains multiple models with different random seeds and averages their
predictions for more robust and stable signals.

Methods:
- Seed ensemble: Same architecture, different weight initializations
- Confidence-weighted voting: Higher confidence models get more weight
"""
import numpy as np
import tensorflow as tf
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.lstm_cnn_attention import build_model


class EnsembleModel:
    def __init__(self, n_models=3, model_dir='models/saved'):
        """
        Args:
            n_models: Number of models in the ensemble
            model_dir: Directory to save individual models
        """
        self.n_models = n_models
        self.model_dir = model_dir
        self.models = []
        self.weights = []  # Per-model confidence weights
    
    def train(self, X_train, y_train, X_val, y_val, input_shape, 
              epochs=50, batch_size=32, ticker="TATASTEEL.NS", **build_kwargs):
        """
        Train N models with different random seeds.
        """
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        self.models = []
        val_accuracies = []
        
        for i in range(self.n_models):
            seed = 42 + i * 100
            print(f"\n{'='*50}")
            print(f"Training Ensemble Model {i+1}/{self.n_models} (seed={seed})")
            print(f"{'='*50}")
            
            # Set seed for reproducibility per model
            tf.random.set_seed(seed)
            np.random.seed(seed)
            
            model = build_model(input_shape=input_shape, **build_kwargs)
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, 
                            restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                patience=5, min_lr=1e-6, verbose=0),
            ]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            loss, acc = model.evaluate(X_val, y_val, verbose=0)
            val_accuracies.append(acc)
            print(f"  Val Accuracy: {acc:.4f}, Val Loss: {loss:.4f}")
            
            self.models.append(model)
            
            # Save individual model
            model_path = os.path.join(self.model_dir, f'{ticker}_ensemble_{i}.keras')
            model.save(model_path)
        
        # Compute confidence weights based on validation accuracy
        total_acc = sum(val_accuracies)
        self.weights = [acc / total_acc for acc in val_accuracies] if total_acc > 0 else [1/self.n_models] * self.n_models
        
        print(f"\nEnsemble Weights: {[f'{w:.3f}' for w in self.weights]}")
        print(f"Mean Val Accuracy: {np.mean(val_accuracies):.4f}")
    
    def predict(self, X):
        """
        Confidence-weighted ensemble prediction.
        Returns averaged probabilities.
        """
        if not self.models:
            raise ValueError("No models trained. Call train() first.")
        
        all_probs = []
        for model, weight in zip(self.models, self.weights):
            probs = model.predict(X, verbose=0)
            all_probs.append(probs * weight)
        
        # Weighted average
        ensemble_probs = np.sum(all_probs, axis=0)
        
        return ensemble_probs
    
    def predict_with_agreement(self, X):
        """
        Returns ensemble probabilities AND agreement score.
        Agreement = fraction of models that predict the same class.
        """
        if not self.models:
            raise ValueError("No models trained. Call train() first.")
        
        all_preds = []
        all_probs = []
        
        for model, weight in zip(self.models, self.weights):
            probs = model.predict(X, verbose=0)
            preds = np.argmax(probs, axis=1)
            all_probs.append(probs * weight)
            all_preds.append(preds)
        
        ensemble_probs = np.sum(all_probs, axis=0)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        # Agreement: what fraction of models agree with the ensemble prediction
        all_preds = np.array(all_preds)  # shape: (n_models, n_samples)
        agreement = np.mean(all_preds == ensemble_preds[np.newaxis, :], axis=0)
        
        return ensemble_probs, agreement
    
    def load(self, ticker="TATASTEEL.NS"):
        """Load saved ensemble models."""
        from tensorflow.keras.models import load_model
        from models.focal_loss import SparseFocalLoss
        
        self.models = []
        for i in range(self.n_models):
            model_path = os.path.join(self.model_dir, f'{ticker}_ensemble_{i}.keras')
            if os.path.exists(model_path):
                model = load_model(model_path, custom_objects={'SparseFocalLoss': SparseFocalLoss})
                self.models.append(model)
        
        if self.models:
            # Equal weights if no training info
            self.weights = [1 / len(self.models)] * len(self.models)
            print(f"Loaded {len(self.models)} ensemble models for {ticker}")
        else:
            print(f"No ensemble models found for {ticker}")


if __name__ == "__main__":
    print("Ensemble model module loaded. Use EnsembleModel class.")
    print("Example:")
    print("  ensemble = EnsembleModel(n_models=3)")
    print("  ensemble.train(X_train, y_train, X_val, y_val, input_shape=(30, 40))")
    print("  probs = ensemble.predict(X_test)")
