"""
LSTM-based Temporal Pattern Analysis Module
Implements sequential attack pattern recognition using LSTM with attention mechanism
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import json
import os
from typing import Tuple, Optional, Dict, List
import pandas as pd


class AttentionLayer(layers.Layer):
    """Self-attention mechanism for LSTM outputs"""
    
    def __init__(self, attention_dim: int = 64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.attention_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.attention_dim,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.attention_dim,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch_size, sequence_length, features)
        uit = tf.nn.tanh(tf.nn.bias_add(tf.tensordot(inputs, self.W, axes=1), self.b))
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait, axis=1)
        
        # Expand dimensions for broadcasting
        ait = tf.expand_dims(ait, axis=-1)
        
        # Apply attention weights
        weighted_input = inputs * ait
        output = tf.reduce_sum(weighted_input, axis=1)
        
        return output
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'attention_dim': self.attention_dim})
        return config


class LSTMTemporalAnalyzer:
    """
    LSTM-based temporal pattern analysis for IDS
    Architecture: Sequence input → LSTM(64) → LSTM(32) → Attention → Dense layers
    """
    
    def __init__(self, input_dim: int = 41, sequence_length: int = 15, num_classes: int = 5):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.model = None
        self.is_trained = False
        
    def build_model(self) -> tf.keras.Model:
        """Build the LSTM architecture with attention mechanism"""
        # Input layer for sequences
        input_layer = keras.Input(
            shape=(self.sequence_length, self.input_dim), 
            name='sequence_input'
        )
        
        # First LSTM layer
        lstm_1 = layers.LSTM(
            64, 
            return_sequences=True, 
            dropout=0.2, 
            recurrent_dropout=0.2,
            name='lstm_1'
        )(input_layer)
        
        lstm_1 = layers.BatchNormalization(name='lstm_bn_1')(lstm_1)
        
        # Second LSTM layer
        lstm_2 = layers.LSTM(
            32, 
            return_sequences=True, 
            dropout=0.2, 
            recurrent_dropout=0.2,
            name='lstm_2'
        )(lstm_1)
        
        lstm_2 = layers.BatchNormalization(name='lstm_bn_2')(lstm_2)
        
        # Attention mechanism
        attention_output = AttentionLayer(
            attention_dim=32, 
            name='attention'
        )(lstm_2)
        
        # Dense layers for classification
        dense_1 = layers.Dense(32, activation='relu', name='dense_1')(attention_output)
        dense_1 = layers.BatchNormalization(name='dense_bn_1')(dense_1)
        dense_1 = layers.Dropout(0.3, name='dense_dropout_1')(dense_1)
        
        dense_2 = layers.Dense(16, activation='relu', name='dense_2')(dense_1)
        dense_2 = layers.BatchNormalization(name='dense_bn_2')(dense_2)
        dense_2 = layers.Dropout(0.2, name='dense_dropout_2')(dense_2)
        
        # Output layer
        output_layer = layers.Dense(
            self.num_classes, 
            activation='softmax', 
            name='output'
        )(dense_2)
        
        # Build the model
        model = keras.Model(input_layer, output_layer, name='lstm_temporal_analyzer')
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_sequences: np.ndarray, y_sequences: np.ndarray,
              validation_data: Optional[Tuple] = None, epochs: int = 100, 
              batch_size: int = 32) -> Dict:
        """
        Train the LSTM model on sequential attack patterns
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', 
                patience=20, 
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=15, 
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'models_advanced/lstm_checkpoint.keras',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        print(f"Training LSTM on {X_sequences.shape[0]} sequences...")
        print(f"Input shape: {X_sequences.shape}")
        print(f"Target shape: {y_sequences.shape}")
        
        # Train the model
        history = self.model.fit(
            X_sequences, y_sequences,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        return {
            'accuracy': history.history['accuracy'][-1],
            'val_accuracy': history.history['val_accuracy'][-1] if validation_data else None,
            'loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1] if validation_data else None,
            'epochs_trained': len(history.history['loss'])
        }
    
    def predict_sequences(self, X_sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict attack types for sequences
        Returns: (predictions, confidence_scores)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("LSTM model must be trained before making predictions")
        
        predictions_proba = self.model.predict(X_sequences, verbose=0)
        predictions = np.argmax(predictions_proba, axis=1)
        confidence_scores = np.max(predictions_proba, axis=1)
        
        return predictions, confidence_scores
    
    def analyze_temporal_patterns(self, X_sequences: np.ndarray) -> Dict:
        """
        Analyze temporal patterns in the sequences
        Returns pattern analysis results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        predictions, confidence_scores = self.predict_sequences(X_sequences)
        
        # Calculate pattern metrics
        pattern_consistency = np.std(confidence_scores)  # Lower std = more consistent
        avg_confidence = np.mean(confidence_scores)
        
        # Analyze sequence transitions
        unique_predictions, counts = np.unique(predictions, return_counts=True)
        pattern_distribution = dict(zip(unique_predictions, counts))
        
        return {
            'predictions': predictions,
            'confidence_scores': confidence_scores,
            'avg_confidence': avg_confidence,
            'pattern_consistency': pattern_consistency,
            'pattern_distribution': pattern_distribution,
            'sequence_length': self.sequence_length
        }
    
    def save_model(self, model_dir: str = "models_advanced"):
        """Save the trained LSTM model"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        if self.model is not None:
            self.model.save(os.path.join(model_dir, "lstm_model.keras"))
        
        # Save model metadata
        metadata = {
            'input_dim': self.input_dim,
            'sequence_length': self.sequence_length,
            'num_classes': self.num_classes,
            'is_trained': self.is_trained
        }
        
        with open(os.path.join(model_dir, "lstm_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"LSTM model saved to {model_dir}/")
    
    def load_model(self, model_dir: str = "models_advanced"):
        """Load a pre-trained LSTM model"""
        try:
            # Load the custom layer
            custom_objects = {'AttentionLayer': AttentionLayer}
            
            self.model = keras.models.load_model(
                os.path.join(model_dir, "lstm_model.keras"),
                custom_objects=custom_objects
            )
            
            # Load metadata
            metadata_path = os.path.join(model_dir, "lstm_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.input_dim = metadata['input_dim']
                self.sequence_length = metadata['sequence_length']
                self.num_classes = metadata['num_classes']
                self.is_trained = metadata['is_trained']
            
            print(f"LSTM model loaded from {model_dir}/")
            return True
            
        except Exception as e:
            print(f"Error loading LSTM model: {str(e)}")
            return False


class SequenceGenerator:
    """
    Generate temporal sequences from NSL-KDD-style data for LSTM training
    """
    
    def __init__(self, sequence_length: int = 15, overlap: float = 0.5):
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.step_size = max(1, int(sequence_length * (1 - overlap)))
        
    def create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create temporal sequences from feature data
        Uses sliding window approach to generate sequences
        """
        if len(features) != len(labels):
            raise ValueError("Features and labels must have the same length")
        
        if len(features) < self.sequence_length:
            raise ValueError(f"Data length ({len(features)}) must be >= sequence_length ({self.sequence_length})")
        
        sequences = []
        sequence_labels = []
        
        # Create sliding windows
        for i in range(0, len(features) - self.sequence_length + 1, self.step_size):
            sequence = features[i:i + self.sequence_length]
            
            # For labels, use the most frequent label in the sequence (majority vote)
            window_labels = labels[i:i + self.sequence_length]
            label = self._get_sequence_label(window_labels)
            
            sequences.append(sequence)
            sequence_labels.append(label)
        
        sequences = np.array(sequences)
        sequence_labels = np.array(sequence_labels)
        
        print(f"Generated {len(sequences)} sequences from {len(features)} samples")
        print(f"Sequence shape: {sequences.shape}")
        
        return sequences, sequence_labels
    
    def _get_sequence_label(self, window_labels: np.ndarray) -> int:
        """
        Determine the label for a sequence window
        Uses majority vote, with preference for attack labels over normal
        """
        unique_labels, counts = np.unique(window_labels, return_counts=True)
        
        # If there are any attacks in the window, prioritize them
        if len(unique_labels) > 1 and 0 in unique_labels:
            # Remove normal class and choose from attacks
            attack_mask = unique_labels != 0
            if np.any(attack_mask):
                attack_labels = unique_labels[attack_mask]
                attack_counts = counts[attack_mask]
                return attack_labels[np.argmax(attack_counts)]
        
        # Otherwise, use simple majority vote
        return unique_labels[np.argmax(counts)]
    
    def create_synthetic_sequences(self, normal_features: np.ndarray, 
                                 attack_features: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic attack sequences by combining normal and attack patterns
        Helps with data augmentation for minority attack classes
        """
        synthetic_sequences = []
        synthetic_labels = []
        
        for attack_class, attack_data in attack_features.items():
            if len(attack_data) == 0:
                continue
                
            # Create sequences with mixed normal and attack patterns
            for _ in range(min(100, len(attack_data) // 5)):  # Generate up to 100 synthetic sequences per class
                sequence = []
                
                # Start with some normal traffic
                normal_start = np.random.randint(0, max(1, len(normal_features) - 5))
                sequence.extend(normal_features[normal_start:normal_start + 3])
                
                # Add attack patterns
                attack_start = np.random.randint(0, max(1, len(attack_data) - 10))
                attack_length = min(self.sequence_length - 5, len(attack_data) - attack_start, 8)
                sequence.extend(attack_data[attack_start:attack_start + attack_length])
                
                # Fill remaining with normal or attack data
                remaining_length = self.sequence_length - len(sequence)
                if remaining_length > 0:
                    if np.random.random() > 0.5 and len(attack_data) > attack_start + attack_length + remaining_length:
                        # Continue with attack data
                        sequence.extend(attack_data[attack_start + attack_length:attack_start + attack_length + remaining_length])
                    else:
                        # Fill with normal data
                        normal_fill = np.random.randint(0, max(1, len(normal_features) - remaining_length))
                        sequence.extend(normal_features[normal_fill:normal_fill + remaining_length])
                
                # Ensure correct length
                if len(sequence) == self.sequence_length:
                    synthetic_sequences.append(np.array(sequence))
                    synthetic_labels.append(attack_class)
        
        if synthetic_sequences:
            synthetic_sequences = np.array(synthetic_sequences)
            synthetic_labels = np.array(synthetic_labels)
            print(f"Generated {len(synthetic_sequences)} synthetic attack sequences")
            return synthetic_sequences, synthetic_labels
        else:
            return np.array([]), np.array([])


if __name__ == "__main__":
    # Example usage and testing
    print("Testing LSTM Temporal Analyzer...")
    
    # Create dummy sequential data
    np.random.seed(42)
    
    # Generate sample sequences
    sequence_length = 15
    input_dim = 41
    num_samples = 1000
    
    X_sequences = np.random.random((num_samples, sequence_length, input_dim))
    y_sequences = np.random.randint(0, 5, (num_samples,))
    
    # Split data
    split_idx = int(0.8 * num_samples)
    X_train, X_val = X_sequences[:split_idx], X_sequences[split_idx:]
    y_train, y_val = y_sequences[:split_idx], y_sequences[split_idx:]
    
    # Initialize and train LSTM
    analyzer = LSTMTemporalAnalyzer(input_dim=input_dim, sequence_length=sequence_length)
    
    # Train the model
    train_results = analyzer.train(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10
    )
    print("Training results:", train_results)
    
    # Test pattern analysis
    pattern_analysis = analyzer.analyze_temporal_patterns(X_val[:50])
    print("Pattern analysis sample:", {k: v for k, v in pattern_analysis.items() if k != 'predictions'})
    
    # Save model
    analyzer.save_model()
    print("LSTM model saved successfully!")
    
    # Test sequence generation
    print("\nTesting sequence generation...")
    seq_generator = SequenceGenerator(sequence_length=10)
    
    # Create sample flat data
    features = np.random.random((200, input_dim))
    labels = np.random.randint(0, 5, (200,))
    
    sequences, seq_labels = seq_generator.create_sequences(features, labels)
    print(f"Generated sequences shape: {sequences.shape}")
    print(f"Generated labels shape: {seq_labels.shape}")