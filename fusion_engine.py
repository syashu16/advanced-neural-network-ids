"""
Hybrid Fusion Engine
Intelligently combines ensemble models, autoencoder, and LSTM predictions
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, List, Optional
import json
import os
import joblib
from dataclasses import dataclass
from anomaly_detector import AutoencoderAnomalyDetector
from temporal_analyzer import LSTMTemporalAnalyzer, SequenceGenerator


@dataclass
class PredictionResults:
    """Container for prediction results from all components"""
    ensemble_prediction: int
    ensemble_confidence: float
    ensemble_probabilities: np.ndarray
    
    anomaly_score: float
    is_anomaly: bool
    
    temporal_prediction: Optional[int] = None
    temporal_confidence: Optional[float] = None
    
    final_prediction: Optional[int] = None
    final_confidence: Optional[float] = None
    decision_reasoning: Optional[str] = None


class HybridFusionEngine:
    """
    Advanced fusion engine combining ensemble, autoencoder, and LSTM predictions
    Uses adaptive thresholds and intelligent decision logic
    """
    
    def __init__(self, ensemble_models: List, autoencoder: AutoencoderAnomalyDetector,
                 lstm_analyzer: Optional[LSTMTemporalAnalyzer] = None):
        self.ensemble_models = ensemble_models
        self.autoencoder = autoencoder
        self.lstm_analyzer = lstm_analyzer
        self.sequence_generator = SequenceGenerator() if lstm_analyzer else None
        
        # Fusion weights (learnable parameters)
        self.fusion_weights = {
            'ensemble_weight': 0.6,      # High weight for proven ensemble
            'anomaly_weight': 0.25,      # Medium weight for anomaly detection
            'temporal_weight': 0.15      # Lower weight for temporal (when available)
        }
        
        # Decision thresholds
        self.thresholds = {
            'high_confidence_threshold': 0.9,    # Direct ensemble decision
            'anomaly_threshold': None,            # Set from autoencoder
            'temporal_consistency_threshold': 0.7,
            'fusion_decision_threshold': 0.5
        }
        
        # Performance tracking
        self.decision_stats = {
            'ensemble_decisions': 0,
            'anomaly_overrides': 0,
            'temporal_influences': 0,
            'fusion_decisions': 0
        }
        
        self.class_names = ["Normal", "DoS", "Probe", "R2L", "U2R"]
        
    def set_anomaly_threshold(self, threshold: float):
        """Set the anomaly detection threshold from the autoencoder"""
        self.thresholds['anomaly_threshold'] = threshold
        
    def update_fusion_weights(self, weights: Dict[str, float]):
        """Update fusion weights based on validation performance"""
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            # Normalize weights
            for key in weights:
                weights[key] /= total_weight
        
        self.fusion_weights.update(weights)
        print(f"Updated fusion weights: {self.fusion_weights}")
    
    def make_ensemble_prediction(self, input_data: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Make prediction using ensemble models"""
        if not self.ensemble_models:
            raise ValueError("No ensemble models available")
        
        predictions = []
        for model in self.ensemble_models:
            pred = model.predict(input_data, verbose=0)
            predictions.append(pred)
        
        # Average ensemble predictions
        ensemble_pred = np.mean(predictions, axis=0)
        pred_class = np.argmax(ensemble_pred)
        confidence = float(ensemble_pred[0][pred_class])
        
        return pred_class, confidence, ensemble_pred[0]
    
    def make_single_prediction(self, input_data: np.ndarray) -> PredictionResults:
        """
        Make prediction for a single input using all available components
        """
        # 1. Ensemble prediction (baseline)
        ensemble_pred, ensemble_conf, ensemble_probs = self.make_ensemble_prediction(input_data)
        
        # 2. Anomaly detection
        anomaly_scores, is_anomaly, _ = self.autoencoder.detect_anomalies(input_data)
        anomaly_score = float(anomaly_scores[0])
        is_anomaly = bool(is_anomaly[0])
        
        # 3. Create results container
        results = PredictionResults(
            ensemble_prediction=ensemble_pred,
            ensemble_confidence=ensemble_conf,
            ensemble_probabilities=ensemble_probs,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly
        )
        
        # 4. Apply fusion logic
        results = self._apply_fusion_logic(results)
        
        return results
    
    def make_sequence_prediction(self, sequence_data: np.ndarray, 
                               individual_predictions: List[PredictionResults]) -> PredictionResults:
        """
        Make prediction for a sequence using LSTM and fusion logic
        """
        if self.lstm_analyzer is None:
            # Fallback to majority vote of individual predictions
            pred_classes = [pred.final_prediction or pred.ensemble_prediction for pred in individual_predictions]
            final_pred = max(set(pred_classes), key=pred_classes.count)
            
            # Average confidence
            confidences = [pred.final_confidence or pred.ensemble_confidence for pred in individual_predictions]
            final_conf = np.mean(confidences)
            
            return PredictionResults(
                ensemble_prediction=final_pred,
                ensemble_confidence=final_conf,
                ensemble_probabilities=individual_predictions[-1].ensemble_probabilities,
                anomaly_score=np.mean([pred.anomaly_score for pred in individual_predictions]),
                is_anomaly=any(pred.is_anomaly for pred in individual_predictions),
                final_prediction=final_pred,
                final_confidence=final_conf,
                decision_reasoning="Sequence majority vote (no LSTM)"
            )
        
        # LSTM temporal prediction
        if sequence_data.ndim == 2:
            sequence_data = sequence_data.reshape(1, *sequence_data.shape)
        
        temporal_preds, temporal_confs = self.lstm_analyzer.predict_sequences(sequence_data)
        temporal_pred = int(temporal_preds[0])
        temporal_conf = float(temporal_confs[0])
        
        # Use the last individual prediction as base
        base_results = individual_predictions[-1]
        base_results.temporal_prediction = temporal_pred
        base_results.temporal_confidence = temporal_conf
        
        # Apply temporal-aware fusion
        results = self._apply_temporal_fusion_logic(base_results, individual_predictions)
        
        return results
    
    def _apply_fusion_logic(self, results: PredictionResults) -> PredictionResults:
        """
        Apply intelligent fusion logic for single predictions
        """
        ensemble_pred = results.ensemble_prediction
        ensemble_conf = results.ensemble_confidence
        is_anomaly = results.is_anomaly
        
        # Decision Logic 1: High confidence ensemble prediction
        if ensemble_conf >= self.thresholds['high_confidence_threshold']:
            if not is_anomaly or ensemble_pred != 0:  # Not anomalous OR predicting attack
                results.final_prediction = ensemble_pred
                results.final_confidence = ensemble_conf
                results.decision_reasoning = f"High confidence ensemble ({ensemble_conf:.3f})"
                self.decision_stats['ensemble_decisions'] += 1
                return results
        
        # Decision Logic 2: Anomaly override for normal predictions
        if is_anomaly and ensemble_pred == 0:
            # Anomaly detected but ensemble says normal - flag as potential novel attack
            # Choose the attack class with highest probability (excluding normal)
            attack_probs = results.ensemble_probabilities[1:]  # Exclude normal class
            if len(attack_probs) > 0:
                attack_class = np.argmax(attack_probs) + 1  # +1 because we excluded normal (0)
                attack_confidence = float(attack_probs[np.argmax(attack_probs)])
                
                # Weighted confidence considering anomaly score
                anomaly_influence = min(results.anomaly_score / (self.thresholds['anomaly_threshold'] or 1.0), 2.0)
                adjusted_confidence = attack_confidence * (1 + 0.3 * anomaly_influence)
                adjusted_confidence = min(adjusted_confidence, 0.95)  # Cap at 95%
                
                results.final_prediction = attack_class
                results.final_confidence = adjusted_confidence
                results.decision_reasoning = f"Anomaly override: score={results.anomaly_score:.3f}"
                self.decision_stats['anomaly_overrides'] += 1
                return results
        
        # Decision Logic 3: Standard ensemble prediction with anomaly consideration
        if is_anomaly:
            # Reduce confidence slightly for anomalous data
            adjusted_conf = ensemble_conf * 0.85
        else:
            # Increase confidence slightly for normal data pattern
            adjusted_conf = min(ensemble_conf * 1.1, 0.99)
        
        results.final_prediction = ensemble_pred
        results.final_confidence = adjusted_conf
        results.decision_reasoning = f"Ensemble with anomaly adjustment ({'anomalous' if is_anomaly else 'normal'} pattern)"
        self.decision_stats['fusion_decisions'] += 1
        
        return results
    
    def _apply_temporal_fusion_logic(self, base_results: PredictionResults, 
                                   sequence_predictions: List[PredictionResults]) -> PredictionResults:
        """
        Apply temporal-aware fusion logic using LSTM predictions
        """
        temporal_pred = base_results.temporal_prediction
        temporal_conf = base_results.temporal_confidence
        ensemble_pred = base_results.ensemble_prediction
        ensemble_conf = base_results.ensemble_confidence
        
        # Calculate sequence consistency
        sequence_anomaly_rate = np.mean([pred.is_anomaly for pred in sequence_predictions])
        sequence_ensemble_consistency = np.std([pred.ensemble_prediction for pred in sequence_predictions])
        
        # Decision Logic 1: High temporal confidence with consistency
        if temporal_conf >= self.thresholds['temporal_consistency_threshold']:
            # Check if temporal prediction aligns with recent patterns
            recent_preds = [pred.final_prediction or pred.ensemble_prediction 
                          for pred in sequence_predictions[-5:]]  # Last 5 predictions
            
            if recent_preds.count(temporal_pred) >= len(recent_preds) * 0.6:  # 60% consistency
                # Combine temporal and ensemble confidences
                fusion_conf = (temporal_conf * self.fusion_weights['temporal_weight'] + 
                             ensemble_conf * self.fusion_weights['ensemble_weight'])
                
                base_results.final_prediction = temporal_pred
                base_results.final_confidence = fusion_conf
                base_results.decision_reasoning = f"Temporal consistency ({temporal_conf:.3f})"
                self.decision_stats['temporal_influences'] += 1
                return base_results
        
        # Decision Logic 2: Persistent anomaly pattern in sequence
        if sequence_anomaly_rate > 0.6:  # >60% of sequence is anomalous
            # Strong indication of attack sequence
            if temporal_pred != 0 and temporal_pred == ensemble_pred:
                # Both agree on attack type
                weighted_conf = (temporal_conf * self.fusion_weights['temporal_weight'] + 
                               ensemble_conf * self.fusion_weights['ensemble_weight'] +
                               sequence_anomaly_rate * self.fusion_weights['anomaly_weight'])
                
                base_results.final_prediction = temporal_pred
                base_results.final_confidence = min(weighted_conf, 0.95)
                base_results.decision_reasoning = f"Persistent anomaly sequence (rate={sequence_anomaly_rate:.2f})"
                self.decision_stats['temporal_influences'] += 1
                return base_results
        
        # Decision Logic 3: Temporal disagreement resolution
        if temporal_pred != ensemble_pred:
            # Weight the predictions based on confidence and consistency
            temporal_weight = temporal_conf * self.fusion_weights['temporal_weight']
            ensemble_weight = ensemble_conf * self.fusion_weights['ensemble_weight']
            
            if ensemble_weight > temporal_weight:
                final_pred = ensemble_pred
                final_conf = ensemble_conf * 0.9  # Slight penalty for disagreement
                reasoning = "Ensemble over temporal (confidence)"
            else:
                final_pred = temporal_pred
                final_conf = temporal_conf * 0.9  # Slight penalty for disagreement
                reasoning = "Temporal over ensemble (confidence)"
            
            base_results.final_prediction = final_pred
            base_results.final_confidence = final_conf
            base_results.decision_reasoning = reasoning
            self.decision_stats['fusion_decisions'] += 1
            return base_results
        
        # Default: use base fusion logic
        return self._apply_fusion_logic(base_results)
    
    def optimize_fusion_weights(self, validation_data: List[Tuple[np.ndarray, int]], 
                              num_iterations: int = 50) -> Dict[str, float]:
        """
        Optimize fusion weights using validation data
        """
        if not validation_data:
            return self.fusion_weights
        
        best_accuracy = 0.0
        best_weights = self.fusion_weights.copy()
        
        print("Optimizing fusion weights...")
        
        for iteration in range(num_iterations):
            # Generate random weight variations
            weight_variation = {
                'ensemble_weight': np.random.uniform(0.4, 0.8),
                'anomaly_weight': np.random.uniform(0.1, 0.4),
                'temporal_weight': np.random.uniform(0.05, 0.25)
            }
            
            # Normalize weights
            total_weight = sum(weight_variation.values())
            for key in weight_variation:
                weight_variation[key] /= total_weight
            
            # Test these weights
            old_weights = self.fusion_weights.copy()
            self.fusion_weights = weight_variation
            
            # Evaluate on validation data
            correct_predictions = 0
            for input_data, true_label in validation_data[:100]:  # Sample for speed
                try:
                    results = self.make_single_prediction(input_data.reshape(1, -1))
                    if results.final_prediction == true_label:
                        correct_predictions += 1
                except:
                    continue
            
            accuracy = correct_predictions / min(100, len(validation_data))
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = weight_variation.copy()
            
            # Restore old weights
            self.fusion_weights = old_weights
        
        # Apply best weights
        self.fusion_weights = best_weights
        print(f"Optimized fusion weights: {self.fusion_weights}")
        print(f"Best validation accuracy: {best_accuracy:.4f}")
        
        return best_weights
    
    def get_decision_statistics(self) -> Dict:
        """Get statistics about decision making patterns"""
        total_decisions = sum(self.decision_stats.values())
        if total_decisions == 0:
            return self.decision_stats
        
        stats = {}
        for decision_type, count in self.decision_stats.items():
            stats[decision_type] = {
                'count': count,
                'percentage': (count / total_decisions) * 100
            }
        
        return stats
    
    def save_fusion_config(self, model_dir: str = "models_advanced"):
        """Save fusion configuration"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        config = {
            'fusion_weights': self.fusion_weights,
            'thresholds': self.thresholds,
            'decision_stats': self.decision_stats,
            'class_names': self.class_names
        }
        
        with open(os.path.join(model_dir, "fusion_config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Fusion configuration saved to {model_dir}/")
    
    def load_fusion_config(self, model_dir: str = "models_advanced"):
        """Load fusion configuration"""
        config_path = os.path.join(model_dir, "fusion_config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.fusion_weights = config.get('fusion_weights', self.fusion_weights)
            self.thresholds = config.get('thresholds', self.thresholds)
            self.decision_stats = config.get('decision_stats', self.decision_stats)
            self.class_names = config.get('class_names', self.class_names)
            
            print(f"Fusion configuration loaded from {model_dir}/")
            return True
        
        return False


def create_hybrid_system(models_dir: str = "models_advanced") -> HybridFusionEngine:
    """
    Factory function to create a complete hybrid system
    """
    print("Creating hybrid fusion system...")
    
    # Load ensemble models
    ensemble_models = []
    for i in range(1, 4):
        model_path = os.path.join(models_dir, f"ensemble_model_{i}.keras")
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            ensemble_models.append(model)
            print(f"Loaded ensemble model {i}")
    
    # Load autoencoder
    autoencoder = AutoencoderAnomalyDetector()
    if autoencoder.load_model(models_dir):
        print("Loaded autoencoder model")
    else:
        print("Warning: Could not load autoencoder model")
    
    # Try to load LSTM
    lstm_analyzer = LSTMTemporalAnalyzer()
    lstm_available = lstm_analyzer.load_model(models_dir)
    if lstm_available:
        print("Loaded LSTM model")
    else:
        print("LSTM model not available - temporal analysis disabled")
        lstm_analyzer = None
    
    # Create fusion engine
    fusion_engine = HybridFusionEngine(
        ensemble_models=ensemble_models,
        autoencoder=autoencoder,
        lstm_analyzer=lstm_analyzer
    )
    
    # Load fusion configuration if available
    fusion_engine.load_fusion_config(models_dir)
    
    # Set anomaly threshold
    if autoencoder.threshold is not None:
        fusion_engine.set_anomaly_threshold(autoencoder.threshold)
    
    print("Hybrid fusion system created successfully!")
    return fusion_engine


if __name__ == "__main__":
    # Test the hybrid fusion system
    print("Testing Hybrid Fusion Engine...")
    
    try:
        # Create hybrid system
        hybrid_system = create_hybrid_system()
        
        # Create test data
        np.random.seed(42)
        test_input = np.random.random((1, 41))
        
        # Make prediction
        results = hybrid_system.make_single_prediction(test_input)
        
        print(f"Prediction results:")
        print(f"  Ensemble: {results.ensemble_prediction} (conf: {results.ensemble_confidence:.3f})")
        print(f"  Anomaly score: {results.anomaly_score:.4f} (anomaly: {results.is_anomaly})")
        print(f"  Final: {results.final_prediction} (conf: {results.final_confidence:.3f})")
        print(f"  Reasoning: {results.decision_reasoning}")
        
        # Show decision statistics
        stats = hybrid_system.get_decision_statistics()
        print(f"Decision statistics: {stats}")
        
    except Exception as e:
        print(f"Error testing hybrid system: {str(e)}")
        print("This is expected if models haven't been trained yet.")