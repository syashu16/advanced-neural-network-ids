#!/usr/bin/env python3
"""
Hybrid System Training Script
Train the autoencoder and LSTM components for the advanced hybrid IDS
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

from hybrid_autoencoder_lstm import HybridAutoEncoderLSTM
from anomaly_detector import AutoencoderAnomalyDetector
from temporal_analyzer import LSTMTemporalAnalyzer, SequenceGenerator


def create_synthetic_training_data():
    """
    Create synthetic training data for demonstration
    In practice, this would load actual NSL-KDD data
    """
    print("Creating synthetic training data...")
    
    np.random.seed(42)
    n_samples = 10000
    n_features = 41
    
    # Create feature data with different patterns for each class
    X_normal = np.random.normal(0, 1, (6000, n_features))  # Normal traffic
    X_dos = np.random.normal(2, 1.5, (2000, n_features))  # DoS attacks  
    X_probe = np.random.normal(-1, 0.8, (1000, n_features))  # Probe attacks
    X_r2l = np.random.normal(1, 2, (600, n_features))  # R2L attacks
    X_u2r = np.random.normal(-2, 1.2, (400, n_features))  # U2R attacks
    
    # Combine all data
    X = np.vstack([X_normal, X_dos, X_probe, X_r2l, X_u2r])
    y = np.hstack([
        np.zeros(6000),  # Normal = 0
        np.ones(2000),   # DoS = 1  
        np.full(1000, 2),  # Probe = 2
        np.full(600, 3),   # R2L = 3
        np.full(400, 4)    # U2R = 4
    ])
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"Created {len(X)} samples with {n_features} features")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    return X, y


def train_autoencoder_component(X, y, models_dir="models_advanced"):
    """Train the autoencoder component"""
    print("\n" + "="*50)
    print("TRAINING AUTOENCODER COMPONENT")
    print("="*50)
    
    # Initialize autoencoder
    autoencoder = AutoencoderAnomalyDetector(input_dim=X.shape[1])
    
    # Extract normal data for training
    normal_mask = y == 0
    normal_data = X[normal_mask]
    print(f"Training on {len(normal_data)} normal samples")
    
    # Train the autoencoder
    train_results = autoencoder.train(
        normal_data,
        epochs=50,  # Reduced for demo
        batch_size=32,
        validation_split=0.2
    )
    
    # Evaluate on test data
    test_normal = normal_data[-500:]  # Last 500 normal samples
    test_anomaly = X[~normal_mask][:500]  # 500 attack samples
    
    performance = autoencoder.evaluate_performance(test_normal, test_anomaly)
    
    print(f"Autoencoder Performance:")
    print(f"  Precision: {performance['precision']:.4f}")
    print(f"  Recall: {performance['recall']:.4f}")  
    print(f"  F1-Score: {performance['f1_score']:.4f}")
    print(f"  Accuracy: {performance['accuracy']:.4f}")
    
    # Save the model
    autoencoder.save_model(models_dir)
    
    return autoencoder, train_results, performance


def train_lstm_component(X, y, sequence_length=15, models_dir="models_advanced"):
    """Train the LSTM component"""
    print("\n" + "="*50)
    print("TRAINING LSTM COMPONENT")
    print("="*50)
    
    # Initialize components
    sequence_generator = SequenceGenerator(sequence_length=sequence_length)
    lstm_analyzer = LSTMTemporalAnalyzer(
        input_dim=X.shape[1],
        sequence_length=sequence_length,
        num_classes=len(np.unique(y))
    )
    
    # Generate sequences
    print("Generating temporal sequences...")
    X_sequences, y_sequences = sequence_generator.create_sequences(X, y)
    
    # Create synthetic sequences for minority classes
    normal_data = X[y == 0]
    attack_data = {}
    for class_id in np.unique(y):
        if class_id != 0:
            attack_data[int(class_id)] = X[y == class_id]
    
    synthetic_X, synthetic_y = sequence_generator.create_synthetic_sequences(
        normal_data, attack_data
    )
    
    # Combine sequences
    if len(synthetic_X) > 0:
        X_sequences = np.concatenate([X_sequences, synthetic_X])
        y_sequences = np.concatenate([y_sequences, synthetic_y])
        print(f"Total sequences after augmentation: {len(X_sequences)}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_sequences, y_sequences, test_size=0.2, random_state=42,
        stratify=y_sequences
    )
    
    # Train LSTM
    train_results = lstm_analyzer.train(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,  # Reduced for demo
        batch_size=32
    )
    
    # Evaluate
    predictions, confidences = lstm_analyzer.predict_sequences(X_val)
    accuracy = np.mean(predictions == y_val)
    
    print(f"LSTM Performance:")
    print(f"  Validation Accuracy: {accuracy:.4f}")
    print(f"  Average Confidence: {np.mean(confidences):.4f}")
    
    # Save the model
    lstm_analyzer.save_model(models_dir)
    
    return lstm_analyzer, sequence_generator, train_results, accuracy


def create_hybrid_system_and_test(X_test, y_test, models_dir="models_advanced"):
    """Create the hybrid system and test it"""
    print("\n" + "="*50) 
    print("CREATING HYBRID SYSTEM")
    print("="*50)
    
    # Initialize hybrid system
    hybrid_system = HybridAutoEncoderLSTM(models_dir)
    
    # Load the hybrid system
    if hybrid_system.load_hybrid_system():
        print("Hybrid system loaded successfully!")
        
        # Get system summary
        summary = hybrid_system.get_model_summary()
        print("System components:")
        for component, status in summary['system_info'].items():
            print(f"  {component}: {status}")
        
        # Test on a few samples
        print("\nTesting hybrid predictions...")
        n_test_samples = min(100, len(X_test))
        
        correct_predictions = 0
        total_anomalies_detected = 0
        
        for i in range(n_test_samples):
            try:
                result = hybrid_system.predict_single(X_test[i:i+1])
                
                if result.final_prediction == y_test[i]:
                    correct_predictions += 1
                
                if result.is_anomaly:
                    total_anomalies_detected += 1
                
                # Show first few predictions
                if i < 5:
                    print(f"  Sample {i}: True={int(y_test[i])}, "
                          f"Pred={result.final_prediction}, "
                          f"Conf={result.final_confidence:.3f}, "
                          f"Anomaly={result.is_anomaly}")
            
            except Exception as e:
                print(f"  Error predicting sample {i}: {str(e)}")
        
        accuracy = correct_predictions / n_test_samples
        anomaly_rate = total_anomalies_detected / n_test_samples
        
        print(f"\nHybrid System Performance:")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Anomaly Detection Rate: {anomaly_rate:.4f}")
        
        # Get decision statistics
        stats = hybrid_system.fusion_engine.get_decision_statistics()
        print(f"  Decision Statistics: {stats}")
        
        return hybrid_system, accuracy
    
    else:
        print("Failed to load hybrid system")
        return None, 0.0


def main():
    """Main training function"""
    print("🚀 HYBRID AUTOENCODER-LSTM IDS TRAINING")
    print("="*60)
    
    # Create models directory
    models_dir = "models_advanced"
    os.makedirs(models_dir, exist_ok=True)
    
    # Create or load training data
    print("Preparing training data...")
    X, y = create_synthetic_training_data()
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train components
    autoencoder, ae_results, ae_performance = train_autoencoder_component(
        X_train, y_train, models_dir
    )
    
    lstm_analyzer, seq_generator, lstm_results, lstm_accuracy = train_lstm_component(
        X_train, y_train, models_dir=models_dir
    )
    
    # Create and test hybrid system
    hybrid_system, hybrid_accuracy = create_hybrid_system_and_test(
        X_test, y_test, models_dir
    )
    
    # Save training summary
    training_summary = {
        'training_completed': True,
        'training_date': pd.Timestamp.now().isoformat(),
        'data_info': {
            'total_samples': len(X),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X.shape[1],
            'classes': len(np.unique(y))
        },
        'autoencoder_results': {
            'training_loss': ae_results['loss'],
            'validation_loss': ae_results['val_loss'],
            'threshold': ae_results['threshold'],
            'epochs': ae_results['epochs_trained'],
            'performance': ae_performance
        },
        'lstm_results': {
            'training_accuracy': lstm_results['accuracy'],
            'validation_accuracy': lstm_results['val_accuracy'],
            'epochs': lstm_results['epochs_trained'],
            'test_accuracy': lstm_accuracy
        },
        'hybrid_system_results': {
            'hybrid_accuracy': hybrid_accuracy,
            'components_active': hybrid_system is not None
        }
    }
    
    with open(os.path.join(models_dir, 'hybrid_training_results.json'), 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📊 Final Results:")
    print(f"  Autoencoder Accuracy: {ae_performance['accuracy']:.4f}")
    print(f"  LSTM Accuracy: {lstm_accuracy:.4f}")  
    print(f"  Hybrid System Accuracy: {hybrid_accuracy:.4f}")
    print(f"\n💾 All models saved to: {models_dir}/")
    print(f"🎯 Ready for deployment in Streamlit app!")


if __name__ == "__main__":
    main()