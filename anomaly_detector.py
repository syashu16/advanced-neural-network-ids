"""
Autoencoder-based Anomaly Detection Module
Implements unsupervised anomaly detection for novel attack identification
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import json
import os
from typing import Tuple, Optional, Dict
import pandas as pd


class AutoencoderAnomalyDetector:
    """
    Autoencoder-based anomaly detection system for IDS
    Architecture: 41 → 32 → 16 → 8 → 16 → 32 → 41
    """
    
    def __init__(self, input_dim: int = 41, latent_dim: int = 8):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model = None
        self.encoder = None
        self.decoder = None
        self.threshold = None
        self.scaler = None
        self.is_trained = False
        
    def build_model(self) -> tf.keras.Model:
        """Build the autoencoder architecture"""
        # Input layer
        input_layer = keras.Input(shape=(self.input_dim,), name='input')
        
        # Encoder
        encoded = layers.Dense(32, activation='relu', name='encoder_dense_1')(input_layer)
        encoded = layers.BatchNormalization(name='encoder_bn_1')(encoded)
        encoded = layers.Dropout(0.2, name='encoder_dropout_1')(encoded)
        
        encoded = layers.Dense(16, activation='relu', name='encoder_dense_2')(encoded)
        encoded = layers.BatchNormalization(name='encoder_bn_2')(encoded)
        encoded = layers.Dropout(0.1, name='encoder_dropout_2')(encoded)
        
        # Bottleneck (latent representation)
        latent = layers.Dense(self.latent_dim, activation='relu', name='latent')(encoded)
        
        # Decoder
        decoded = layers.Dense(16, activation='relu', name='decoder_dense_1')(latent)
        decoded = layers.BatchNormalization(name='decoder_bn_1')(decoded)
        decoded = layers.Dropout(0.1, name='decoder_dropout_1')(decoded)
        
        decoded = layers.Dense(32, activation='relu', name='decoder_dense_2')(decoded)
        decoded = layers.BatchNormalization(name='decoder_bn_2')(decoded)
        decoded = layers.Dropout(0.2, name='decoder_dropout_2')(decoded)
        
        # Output layer (reconstruction)
        output_layer = layers.Dense(self.input_dim, activation='sigmoid', name='output')(decoded)
        
        # Build the complete autoencoder
        autoencoder = keras.Model(input_layer, output_layer, name='autoencoder')
        
        # Build encoder separately
        self.encoder = keras.Model(input_layer, latent, name='encoder')
        
        # Build decoder separately 
        decoder_input = keras.Input(shape=(self.latent_dim,))
        
        # Reconstruct decoder architecture
        decoder_decoded = layers.Dense(16, activation='relu', name='decoder_dense_1_standalone')(decoder_input)
        decoder_decoded = layers.BatchNormalization(name='decoder_bn_1_standalone')(decoder_decoded)
        decoder_decoded = layers.Dropout(0.1, name='decoder_dropout_1_standalone')(decoder_decoded)
        
        decoder_decoded = layers.Dense(32, activation='relu', name='decoder_dense_2_standalone')(decoder_decoded)
        decoder_decoded = layers.BatchNormalization(name='decoder_bn_2_standalone')(decoder_decoded)
        decoder_decoded = layers.Dropout(0.2, name='decoder_dropout_2_standalone')(decoder_decoded)
        
        decoder_output = layers.Dense(self.input_dim, activation='sigmoid', name='decoder_output_standalone')(decoder_decoded)
        
        self.decoder = keras.Model(decoder_input, decoder_output, name='decoder')
        
        # Compile the autoencoder
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = autoencoder
        return autoencoder
    
    def train(self, normal_data: np.ndarray, validation_split: float = 0.2, 
              epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Train the autoencoder on normal traffic only (unsupervised learning)
        """
        if self.model is None:
            self.build_model()
        
        # Add early stopping and learning rate reduction
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
            )
        ]
        
        print(f"Training autoencoder on {normal_data.shape[0]} normal samples...")
        
        # Train the autoencoder
        history = self.model.fit(
            normal_data, normal_data,  # Autoencoder learns to reconstruct input
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate threshold based on reconstruction error of normal data
        self._calculate_threshold(normal_data)
        
        self.is_trained = True
        
        return {
            'loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1],
            'threshold': self.threshold,
            'epochs_trained': len(history.history['loss'])
        }
    
    def _calculate_threshold(self, normal_data: np.ndarray, percentile: float = 95.0):
        """Calculate anomaly threshold based on reconstruction error distribution"""
        reconstructions = self.model.predict(normal_data, verbose=0)
        mse_errors = np.mean(np.square(normal_data - reconstructions), axis=1)
        
        # Use 95th percentile as threshold (configurable)
        self.threshold = np.percentile(mse_errors, percentile)
        
        print(f"Anomaly threshold set to: {self.threshold:.6f} ({percentile}th percentile)")
        
        return self.threshold
    
    def detect_anomalies(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect anomalies in the input data
        Returns: (anomaly_scores, is_anomaly, reconstructions)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Autoencoder must be trained before detecting anomalies")
        
        # Get reconstructions
        reconstructions = self.model.predict(data, verbose=0)
        
        # Calculate reconstruction errors (anomaly scores)
        anomaly_scores = np.mean(np.square(data - reconstructions), axis=1)
        
        # Determine anomalies based on threshold
        is_anomaly = anomaly_scores > self.threshold
        
        return anomaly_scores, is_anomaly, reconstructions
    
    def get_latent_representation(self, data: np.ndarray) -> np.ndarray:
        """Get latent representation of input data"""
        if self.encoder is None:
            raise ValueError("Encoder not available. Train the model first.")
        
        return self.encoder.predict(data, verbose=0)
    
    def save_model(self, model_dir: str = "models_advanced"):
        """Save the trained autoencoder and threshold"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        if self.model is not None:
            self.model.save(os.path.join(model_dir, "autoencoder_model.keras"))
        
        if self.encoder is not None:
            self.encoder.save(os.path.join(model_dir, "encoder_model.keras"))
        
        if self.decoder is not None:
            self.decoder.save(os.path.join(model_dir, "decoder_model.keras"))
        
        # Save threshold and metadata
        threshold_data = {
            'threshold': float(self.threshold) if self.threshold else None,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'is_trained': self.is_trained
        }
        
        with open(os.path.join(model_dir, "anomaly_threshold.json"), 'w') as f:
            json.dump(threshold_data, f, indent=2)
        
        print(f"Autoencoder models saved to {model_dir}/")
    
    def load_model(self, model_dir: str = "models_advanced"):
        """Load a pre-trained autoencoder"""
        try:
            self.model = keras.models.load_model(
                os.path.join(model_dir, "autoencoder_model.keras")
            )
            
            if os.path.exists(os.path.join(model_dir, "encoder_model.keras")):
                self.encoder = keras.models.load_model(
                    os.path.join(model_dir, "encoder_model.keras")
                )
            
            if os.path.exists(os.path.join(model_dir, "decoder_model.keras")):
                self.decoder = keras.models.load_model(
                    os.path.join(model_dir, "decoder_model.keras")
                )
            
            # Load threshold
            threshold_path = os.path.join(model_dir, "anomaly_threshold.json")
            if os.path.exists(threshold_path):
                with open(threshold_path, 'r') as f:
                    threshold_data = json.load(f)
                
                self.threshold = threshold_data['threshold']
                self.input_dim = threshold_data['input_dim']
                self.latent_dim = threshold_data['latent_dim']
                self.is_trained = threshold_data['is_trained']
            
            print(f"Autoencoder loaded from {model_dir}/")
            return True
            
        except Exception as e:
            print(f"Error loading autoencoder: {str(e)}")
            return False
    
    def evaluate_performance(self, normal_data: np.ndarray, 
                           anomaly_data: np.ndarray) -> Dict:
        """Evaluate autoencoder performance on normal vs anomaly data"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Get anomaly scores
        normal_scores, normal_pred, _ = self.detect_anomalies(normal_data)
        anomaly_scores, anomaly_pred, _ = self.detect_anomalies(anomaly_data)
        
        # Calculate metrics
        true_negatives = np.sum(~normal_pred)  # Normal data correctly classified
        false_positives = np.sum(normal_pred)  # Normal data incorrectly flagged
        true_positives = np.sum(anomaly_pred)  # Anomaly data correctly detected
        false_negatives = np.sum(~anomaly_pred)  # Anomaly data missed
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        accuracy = (true_positives + true_negatives) / (len(normal_data) + len(anomaly_data))
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'threshold': self.threshold,
            'normal_score_mean': np.mean(normal_scores),
            'normal_score_std': np.std(normal_scores),
            'anomaly_score_mean': np.mean(anomaly_scores),
            'anomaly_score_std': np.std(anomaly_scores)
        }


def create_training_data_from_ensemble_predictions(feature_data: np.ndarray, 
                                                 predictions: np.ndarray) -> np.ndarray:
    """Create normal traffic data for autoencoder training from ensemble predictions"""
    # Extract only the data points predicted as normal (class 0)
    normal_mask = predictions == 0
    normal_data = feature_data[normal_mask]
    
    print(f"Extracted {normal_data.shape[0]} normal samples for autoencoder training")
    return normal_data


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Autoencoder Anomaly Detector...")
    
    # Create dummy data for testing
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (1000, 41))
    anomaly_data = np.random.normal(3, 1, (100, 41))  # Different distribution
    
    # Initialize and train autoencoder
    detector = AutoencoderAnomalyDetector(input_dim=41)
    
    # Train on normal data
    train_results = detector.train(normal_data, epochs=50)
    print("Training results:", train_results)
    
    # Test anomaly detection
    performance = detector.evaluate_performance(normal_data, anomaly_data)
    print("Performance metrics:", performance)
    
    # Save model
    detector.save_model()
    print("Model saved successfully!")