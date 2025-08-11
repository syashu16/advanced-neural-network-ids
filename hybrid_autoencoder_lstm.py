"""
Hybrid Autoencoder-LSTM IDS Implementation
Main orchestrator for the advanced hybrid neural network IDS system
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from anomaly_detector import AutoencoderAnomalyDetector, create_training_data_from_ensemble_predictions
from temporal_analyzer import LSTMTemporalAnalyzer, SequenceGenerator
from fusion_engine import HybridFusionEngine, create_hybrid_system, PredictionResults


class HybridAutoEncoderLSTM:
    """
    Main class for the Hybrid Autoencoder-LSTM IDS system
    Coordinates training, evaluation, and inference across all components
    """
    
    def __init__(self, models_dir: str = "models_advanced"):
        self.models_dir = models_dir
        self.fusion_engine = None
        
        # Component models
        self.ensemble_models = []
        self.autoencoder = None
        self.lstm_analyzer = None
        self.sequence_generator = None
        
        # Data preprocessing
        self.scaler = None
        self.feature_columns = None
        self.label_encoders = None
        
        # Training history and metrics
        self.training_history = {}
        self.performance_metrics = {}
        
    def load_existing_models(self) -> bool:
        """Load existing ensemble models and preprocessing components"""
        try:
            # Load preprocessing components
            self.scaler = joblib.load(os.path.join(self.models_dir, 'scaler_advanced.pkl'))
            self.feature_columns = joblib.load(os.path.join(self.models_dir, 'feature_columns.pkl'))
            self.label_encoders = joblib.load(os.path.join(self.models_dir, 'label_encoders.pkl'))
            
            # Load ensemble models
            for i in range(1, 4):
                model_path = os.path.join(self.models_dir, f'ensemble_model_{i}.keras')
                if os.path.exists(model_path):
                    model = tf.keras.models.load_model(model_path)
                    self.ensemble_models.append(model)
            
            print(f"Loaded {len(self.ensemble_models)} ensemble models")
            return len(self.ensemble_models) > 0
            
        except Exception as e:
            print(f"Error loading existing models: {str(e)}")
            return False
    
    def train_autoencoder(self, training_data: np.ndarray, labels: np.ndarray,
                         epochs: int = 100, batch_size: int = 32) -> Dict:
        """Train the autoencoder component on normal traffic data"""
        print("Training Autoencoder for Anomaly Detection...")
        
        # Extract normal traffic (label 0) for unsupervised training
        normal_mask = labels == 0
        normal_data = training_data[normal_mask]
        
        print(f"Using {len(normal_data)} normal samples for autoencoder training")
        
        # Initialize autoencoder
        self.autoencoder = AutoencoderAnomalyDetector(input_dim=training_data.shape[1])
        
        # Train the autoencoder
        train_results = self.autoencoder.train(
            normal_data, 
            epochs=epochs, 
            batch_size=batch_size
        )
        
        # Save the trained model
        self.autoencoder.save_model(self.models_dir)
        
        self.training_history['autoencoder'] = train_results
        
        print("Autoencoder training completed!")
        return train_results
    
    def train_lstm(self, training_data: np.ndarray, labels: np.ndarray,
                  sequence_length: int = 15, epochs: int = 100, batch_size: int = 32) -> Dict:
        """Train the LSTM component for temporal pattern recognition"""
        print("Training LSTM for Temporal Pattern Recognition...")
        
        # Initialize components
        self.sequence_generator = SequenceGenerator(sequence_length=sequence_length)
        self.lstm_analyzer = LSTMTemporalAnalyzer(
            input_dim=training_data.shape[1],
            sequence_length=sequence_length,
            num_classes=len(np.unique(labels))
        )
        
        # Generate sequences from the data
        print("Generating temporal sequences...")
        X_sequences, y_sequences = self.sequence_generator.create_sequences(training_data, labels)
        
        # Create synthetic sequences for minority classes
        print("Creating synthetic attack sequences...")
        normal_data = training_data[labels == 0]
        attack_data = {}
        for class_id in np.unique(labels):
            if class_id != 0:
                attack_data[class_id] = training_data[labels == class_id]
        
        synthetic_X, synthetic_y = self.sequence_generator.create_synthetic_sequences(
            normal_data, attack_data
        )
        
        # Combine real and synthetic sequences
        if len(synthetic_X) > 0:
            X_sequences = np.concatenate([X_sequences, synthetic_X])
            y_sequences = np.concatenate([y_sequences, synthetic_y])
            print(f"Total sequences after augmentation: {len(X_sequences)}")
        
        # Split for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_sequences, y_sequences, test_size=0.2, random_state=42, stratify=y_sequences
        )
        
        # Train the LSTM
        train_results = self.lstm_analyzer.train(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Save the trained model
        self.lstm_analyzer.save_model(self.models_dir)
        
        self.training_history['lstm'] = train_results
        
        print("LSTM training completed!")
        return train_results
    
    def train_hybrid_system(self, X_train: np.ndarray, y_train: np.ndarray,
                          autoencoder_epochs: int = 100, lstm_epochs: int = 100,
                          sequence_length: int = 15) -> Dict:
        """Train the complete hybrid system"""
        print("Starting Hybrid System Training...")
        
        # Ensure existing models are loaded
        if not self.ensemble_models:
            if not self.load_existing_models():
                raise ValueError("Cannot load existing ensemble models. Train ensemble first.")
        
        # Stage 1: Train Autoencoder
        autoencoder_results = self.train_autoencoder(
            X_train, y_train, epochs=autoencoder_epochs
        )
        
        # Stage 2: Train LSTM
        lstm_results = self.train_lstm(
            X_train, y_train, sequence_length=sequence_length, epochs=lstm_epochs
        )
        
        # Stage 3: Create and optimize fusion system
        print("Creating fusion system...")
        self.fusion_engine = HybridFusionEngine(
            ensemble_models=self.ensemble_models,
            autoencoder=self.autoencoder,
            lstm_analyzer=self.lstm_analyzer
        )
        
        # Set anomaly threshold
        if self.autoencoder.threshold is not None:
            self.fusion_engine.set_anomaly_threshold(self.autoencoder.threshold)
        
        # Save fusion configuration
        self.fusion_engine.save_fusion_config(self.models_dir)
        
        # Combine training results
        training_summary = {
            'autoencoder': autoencoder_results,
            'lstm': lstm_results,
            'fusion_weights': self.fusion_engine.fusion_weights,
            'training_completed': True
        }
        
        # Save training summary
        with open(os.path.join(self.models_dir, 'hybrid_training_summary.json'), 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        print("Hybrid system training completed!")
        return training_summary
    
    def load_hybrid_system(self) -> bool:
        """Load the complete hybrid system"""
        try:
            # Load existing models
            if not self.load_existing_models():
                return False
            
            # Create fusion system
            self.fusion_engine = create_hybrid_system(self.models_dir)
            
            # Load individual components for direct access
            self.autoencoder = self.fusion_engine.autoencoder
            self.lstm_analyzer = self.fusion_engine.lstm_analyzer
            
            # Create sequence generator if LSTM is available
            if self.lstm_analyzer:
                self.sequence_generator = SequenceGenerator()
            
            print("Hybrid system loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading hybrid system: {str(e)}")
            return False
    
    def predict_single(self, input_data: np.ndarray) -> PredictionResults:
        """Make prediction for a single input"""
        if self.fusion_engine is None:
            raise ValueError("Hybrid system not loaded. Call load_hybrid_system() first.")
        
        return self.fusion_engine.make_single_prediction(input_data)
    
    def predict_sequence(self, sequence_data: np.ndarray) -> PredictionResults:
        """Make prediction for a sequence"""
        if self.fusion_engine is None:
            raise ValueError("Hybrid system not loaded. Call load_hybrid_system() first.")
        
        if sequence_data.ndim == 2:
            # Single sequence
            individual_preds = []
            for i in range(len(sequence_data)):
                pred = self.predict_single(sequence_data[i:i+1])
                individual_preds.append(pred)
            
            return self.fusion_engine.make_sequence_prediction(sequence_data, individual_preds)
        else:
            raise ValueError("Sequence data must be 2D (sequence_length, features)")
    
    def evaluate_on_test_data(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Comprehensive evaluation of the hybrid system"""
        if self.fusion_engine is None:
            raise ValueError("Hybrid system not loaded.")
        
        print("Evaluating hybrid system...")
        
        # Single predictions
        predictions = []
        confidences = []
        anomaly_scores = []
        decision_types = []
        
        for i, sample in enumerate(X_test):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(X_test)} samples")
            
            try:
                result = self.predict_single(sample.reshape(1, -1))
                predictions.append(result.final_prediction)
                confidences.append(result.final_confidence)
                anomaly_scores.append(result.anomaly_score)
                decision_types.append(result.decision_reasoning)
            except Exception as e:
                print(f"Error predicting sample {i}: {str(e)}")
                predictions.append(0)  # Default to normal
                confidences.append(0.5)
                anomaly_scores.append(0.0)
                decision_types.append("Error")
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        accuracy = np.mean(predictions == y_test)
        
        # Per-class metrics
        class_metrics = {}
        for class_id in np.unique(y_test):
            mask = y_test == class_id
            if np.sum(mask) > 0:
                class_predictions = predictions[mask]
                class_accuracy = np.mean(class_predictions == class_id)
                class_metrics[int(class_id)] = {
                    'accuracy': class_accuracy,
                    'samples': int(np.sum(mask)),
                    'correct': int(np.sum(class_predictions == class_id))
                }
        
        # Anomaly detection metrics
        normal_mask = y_test == 0
        attack_mask = y_test != 0
        
        normal_anomaly_scores = np.array(anomaly_scores)[normal_mask]
        attack_anomaly_scores = np.array(anomaly_scores)[attack_mask]
        
        # Decision type analysis
        decision_type_counts = {}
        for decision_type in decision_types:
            if decision_type not in decision_type_counts:
                decision_type_counts[decision_type] = 0
            decision_type_counts[decision_type] += 1
        
        evaluation_results = {
            'overall_accuracy': accuracy,
            'class_metrics': class_metrics,
            'anomaly_detection': {
                'normal_score_mean': float(np.mean(normal_anomaly_scores)) if len(normal_anomaly_scores) > 0 else 0,
                'normal_score_std': float(np.std(normal_anomaly_scores)) if len(normal_anomaly_scores) > 0 else 0,
                'attack_score_mean': float(np.mean(attack_anomaly_scores)) if len(attack_anomaly_scores) > 0 else 0,
                'attack_score_std': float(np.std(attack_anomaly_scores)) if len(attack_anomaly_scores) > 0 else 0,
                'threshold': float(self.autoencoder.threshold) if self.autoencoder.threshold else 0
            },
            'decision_analysis': decision_type_counts,
            'fusion_statistics': self.fusion_engine.get_decision_statistics()
        }
        
        # Save evaluation results
        with open(os.path.join(self.models_dir, 'hybrid_evaluation_results.json'), 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"Evaluation completed! Overall accuracy: {accuracy:.4f}")
        return evaluation_results
    
    def get_model_summary(self) -> Dict:
        """Get comprehensive summary of the hybrid system"""
        summary = {
            'system_info': {
                'ensemble_models': len(self.ensemble_models),
                'autoencoder_available': self.autoencoder is not None and self.autoencoder.is_trained,
                'lstm_available': self.lstm_analyzer is not None and self.lstm_analyzer.is_trained,
                'fusion_engine_ready': self.fusion_engine is not None
            },
            'model_parameters': {}
        }
        
        if self.autoencoder:
            summary['model_parameters']['autoencoder'] = {
                'input_dim': self.autoencoder.input_dim,
                'latent_dim': self.autoencoder.latent_dim,
                'threshold': self.autoencoder.threshold,
                'is_trained': self.autoencoder.is_trained
            }
        
        if self.lstm_analyzer:
            summary['model_parameters']['lstm'] = {
                'input_dim': self.lstm_analyzer.input_dim,
                'sequence_length': self.lstm_analyzer.sequence_length,
                'num_classes': self.lstm_analyzer.num_classes,
                'is_trained': self.lstm_analyzer.is_trained
            }
        
        if self.fusion_engine:
            summary['fusion_config'] = {
                'fusion_weights': self.fusion_engine.fusion_weights,
                'thresholds': self.fusion_engine.thresholds,
                'decision_stats': self.fusion_engine.get_decision_statistics()
            }
        
        return summary


def create_and_train_hybrid_system(training_data_path: Optional[str] = None) -> HybridAutoEncoderLSTM:
    """
    Factory function to create and train a complete hybrid system
    """
    print("Creating and training hybrid system...")
    
    # Initialize hybrid system
    hybrid_system = HybridAutoEncoderLSTM()
    
    if training_data_path and os.path.exists(training_data_path):
        # Load and preprocess training data
        print(f"Loading training data from {training_data_path}")
        # This would be implemented based on your specific data format
        # For now, we'll assume preprocessing is done externally
    
    # Load existing ensemble models
    if not hybrid_system.load_existing_models():
        raise ValueError("Cannot proceed without existing ensemble models")
    
    print("Hybrid system ready for training!")
    return hybrid_system


if __name__ == "__main__":
    # Example usage
    print("Testing Hybrid Autoencoder-LSTM System...")
    
    try:
        # Create hybrid system
        hybrid_system = HybridAutoEncoderLSTM()
        
        # Try to load existing system
        if hybrid_system.load_hybrid_system():
            print("Loaded existing hybrid system")
            
            # Get system summary
            summary = hybrid_system.get_model_summary()
            print("System Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
            
            # Test with dummy data
            test_input = np.random.random((1, 41))
            result = hybrid_system.predict_single(test_input)
            
            print(f"\nTest Prediction:")
            print(f"  Final prediction: {result.final_prediction}")
            print(f"  Confidence: {result.final_confidence:.3f}")
            print(f"  Reasoning: {result.decision_reasoning}")
        
        else:
            print("Hybrid system not available - models need to be trained first")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print("This is expected if the hybrid system hasn't been trained yet")