"""
Real-time Stream Processing Module
Implements real-time network flow analysis with buffering and incremental learning
"""

import numpy as np
import pandas as pd
from collections import deque
import threading
import time
import json
import os
from typing import Dict, List, Callable, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from hybrid_autoencoder_lstm import HybridAutoEncoderLSTM
from fusion_engine import PredictionResults


class NetworkFlowBuffer:
    """
    Circular buffer for maintaining sliding windows of network flows
    """
    
    def __init__(self, max_size: int = 1000, sequence_length: int = 15):
        self.max_size = max_size
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
    def add_flow(self, flow_data: np.ndarray, timestamp: Optional[datetime] = None):
        """Add a new network flow to the buffer"""
        if timestamp is None:
            timestamp = datetime.now()
        
        with self.lock:
            self.buffer.append(flow_data)
            self.timestamps.append(timestamp)
    
    def get_latest_sequence(self) -> Optional[np.ndarray]:
        """Get the most recent sequence for LSTM analysis"""
        with self.lock:
            if len(self.buffer) < self.sequence_length:
                return None
            
            # Get the last sequence_length items
            sequence = list(self.buffer)[-self.sequence_length:]
            return np.array(sequence)
    
    def get_buffer_stats(self) -> Dict:
        """Get statistics about the buffer"""
        with self.lock:
            return {
                'current_size': len(self.buffer),
                'max_size': self.max_size,
                'sequence_length': self.sequence_length,
                'oldest_timestamp': self.timestamps[0] if self.timestamps else None,
                'newest_timestamp': self.timestamps[-1] if self.timestamps else None
            }
    
    def clear_old_entries(self, max_age_minutes: int = 60):
        """Remove entries older than specified age"""
        if not self.timestamps:
            return
        
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        
        with self.lock:
            while self.timestamps and self.timestamps[0] < cutoff_time:
                self.timestamps.popleft()
                self.buffer.popleft()


class AlertManager:
    """
    Manages security alerts and notifications
    """
    
    def __init__(self, alert_threshold: float = 0.8, max_alerts: int = 1000):
        self.alert_threshold = alert_threshold
        self.max_alerts = max_alerts
        self.alerts = deque(maxlen=max_alerts)
        self.alert_callbacks = []
        self.lock = threading.Lock()
        
        # Alert statistics
        self.alert_stats = {
            'total_alerts': 0,
            'alerts_by_type': {},
            'high_confidence_alerts': 0,
            'anomaly_alerts': 0
        }
    
    def add_alert_callback(self, callback: Callable):
        """Add a callback function to be called when alerts are generated"""
        self.alert_callbacks.append(callback)
    
    def process_prediction_result(self, result: PredictionResults, flow_data: np.ndarray):
        """Process a prediction result and generate alerts if necessary"""
        timestamp = datetime.now()
        
        # Determine if this should generate an alert
        should_alert = False
        alert_level = "INFO"
        alert_reason = []
        
        # High confidence attack detection
        if (result.final_prediction != 0 and 
            result.final_confidence >= self.alert_threshold):
            should_alert = True
            alert_level = "HIGH"
            alert_reason.append(f"High confidence attack: {result.final_prediction}")
            self.alert_stats['high_confidence_alerts'] += 1
        
        # Anomaly detection
        if result.is_anomaly and result.anomaly_score > (result.final_confidence * 1.5):
            should_alert = True
            alert_level = "MEDIUM" if alert_level == "INFO" else alert_level
            alert_reason.append(f"Anomaly detected: score={result.anomaly_score:.4f}")
            self.alert_stats['anomaly_alerts'] += 1
        
        # Moderate confidence attack with anomaly
        if (result.final_prediction != 0 and 
            result.final_confidence >= 0.6 and 
            result.is_anomaly):
            should_alert = True
            alert_level = "MEDIUM" if alert_level == "INFO" else alert_level
            alert_reason.append("Combined attack and anomaly indicators")
        
        if should_alert:
            alert = {
                'timestamp': timestamp.isoformat(),
                'level': alert_level,
                'predicted_class': result.final_prediction,
                'confidence': result.final_confidence,
                'anomaly_score': result.anomaly_score,
                'is_anomaly': result.is_anomaly,
                'decision_reasoning': result.decision_reasoning,
                'alert_reasons': alert_reason,
                'flow_summary': {
                    'min': float(np.min(flow_data)),
                    'max': float(np.max(flow_data)),
                    'mean': float(np.mean(flow_data)),
                    'std': float(np.std(flow_data))
                }
            }
            
            with self.lock:
                self.alerts.append(alert)
                self.alert_stats['total_alerts'] += 1
                
                attack_type = result.final_prediction
                if attack_type not in self.alert_stats['alerts_by_type']:
                    self.alert_stats['alerts_by_type'][attack_type] = 0
                self.alert_stats['alerts_by_type'][attack_type] += 1
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    print(f"Alert callback error: {str(e)}")
    
    def get_recent_alerts(self, hours: int = 1) -> List[Dict]:
        """Get alerts from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_alerts = []
            for alert in self.alerts:
                alert_time = datetime.fromisoformat(alert['timestamp'])
                if alert_time >= cutoff_time:
                    recent_alerts.append(alert)
            
            return recent_alerts
    
    def get_alert_statistics(self) -> Dict:
        """Get comprehensive alert statistics"""
        with self.lock:
            stats = self.alert_stats.copy()
            stats['current_alert_count'] = len(self.alerts)
            return stats


class PerformanceMonitor:
    """
    Monitors system performance metrics in real-time
    """
    
    def __init__(self):
        self.metrics = {
            'predictions_per_second': 0.0,
            'average_latency_ms': 0.0,
            'accuracy_estimate': 0.0,
            'memory_usage_mb': 0.0,
            'uptime_hours': 0.0
        }
        
        self.prediction_times = deque(maxlen=100)
        self.latencies = deque(maxlen=100)
        self.recent_predictions = deque(maxlen=1000)
        self.start_time = datetime.now()
        
        self.lock = threading.Lock()
    
    def record_prediction(self, latency_ms: float, prediction_correct: Optional[bool] = None):
        """Record a prediction event for performance tracking"""
        current_time = time.time()
        
        with self.lock:
            self.prediction_times.append(current_time)
            self.latencies.append(latency_ms)
            
            if prediction_correct is not None:
                self.recent_predictions.append(prediction_correct)
            
            self._update_metrics()
    
    def _update_metrics(self):
        """Update performance metrics"""
        current_time = time.time()
        
        # Predictions per second (last 60 seconds)
        recent_times = [t for t in self.prediction_times if current_time - t <= 60]
        self.metrics['predictions_per_second'] = len(recent_times) / 60.0
        
        # Average latency
        if self.latencies:
            self.metrics['average_latency_ms'] = sum(self.latencies) / len(self.latencies)
        
        # Accuracy estimate
        if self.recent_predictions:
            correct_count = sum(1 for pred in self.recent_predictions if pred)
            self.metrics['accuracy_estimate'] = correct_count / len(self.recent_predictions)
        
        # Uptime
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        self.metrics['uptime_hours'] = uptime_seconds / 3600.0
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        with self.lock:
            return self.metrics.copy()


class StreamingProcessor:
    """
    Main streaming processor for real-time IDS analysis
    """
    
    def __init__(self, models_dir: str = "models_advanced", 
                 buffer_size: int = 1000, sequence_length: int = 15):
        # Core components
        self.hybrid_system = HybridAutoEncoderLSTM(models_dir)
        self.flow_buffer = NetworkFlowBuffer(buffer_size, sequence_length)
        self.alert_manager = AlertManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Processing settings
        self.sequence_length = sequence_length
        self.processing_enabled = False
        self.processing_thread = None
        
        # Statistics
        self.stats = {
            'flows_processed': 0,
            'sequences_analyzed': 0,
            'normal_flows': 0,
            'attack_flows': 0,
            'anomalous_flows': 0
        }
        
        self.class_names = ["Normal", "DoS", "Probe", "R2L", "U2R"]
        
        # Load the hybrid system
        if not self.hybrid_system.load_hybrid_system():
            raise ValueError("Could not load hybrid system for streaming")
        
        print("Streaming processor initialized successfully!")
    
    def start_processing(self):
        """Start the real-time processing thread"""
        if self.processing_enabled:
            print("Processing already started")
            return
        
        self.processing_enabled = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("Real-time processing started")
    
    def stop_processing(self):
        """Stop the real-time processing"""
        self.processing_enabled = False
        if self.processing_thread:
            self.processing_thread.join()
        
        print("Real-time processing stopped")
    
    def process_flow(self, flow_data: np.ndarray) -> PredictionResults:
        """Process a single network flow"""
        start_time = time.time()
        
        # Add to buffer
        self.flow_buffer.add_flow(flow_data)
        
        # Make prediction
        result = self.hybrid_system.predict_single(flow_data.reshape(1, -1))
        
        # Update statistics
        self.stats['flows_processed'] += 1
        if result.final_prediction == 0:
            self.stats['normal_flows'] += 1
        else:
            self.stats['attack_flows'] += 1
        
        if result.is_anomaly:
            self.stats['anomalous_flows'] += 1
        
        # Process alerts
        self.alert_manager.process_prediction_result(result, flow_data)
        
        # Record performance
        latency_ms = (time.time() - start_time) * 1000
        self.performance_monitor.record_prediction(latency_ms)
        
        return result
    
    def process_sequence(self, sequence_data: Optional[np.ndarray] = None) -> Optional[PredictionResults]:
        """Process a sequence using LSTM analysis"""
        if sequence_data is None:
            sequence_data = self.flow_buffer.get_latest_sequence()
        
        if sequence_data is None:
            return None
        
        start_time = time.time()
        
        # Make sequence prediction
        result = self.hybrid_system.predict_sequence(sequence_data)
        
        # Update statistics
        self.stats['sequences_analyzed'] += 1
        
        # Record performance
        latency_ms = (time.time() - start_time) * 1000
        self.performance_monitor.record_prediction(latency_ms)
        
        return result
    
    def _processing_loop(self):
        """Main processing loop for continuous analysis"""
        print("Starting continuous processing loop...")
        
        while self.processing_enabled:
            try:
                # Check if we have enough data for sequence analysis
                if len(self.flow_buffer.buffer) >= self.sequence_length:
                    # Periodic sequence analysis (every 10 seconds)
                    sequence_result = self.process_sequence()
                    if sequence_result:
                        self.alert_manager.process_prediction_result(
                            sequence_result, 
                            self.flow_buffer.get_latest_sequence()[-1]
                        )
                
                # Clean up old buffer entries
                self.flow_buffer.clear_old_entries(max_age_minutes=30)
                
                # Sleep to prevent excessive CPU usage
                time.sleep(10)  # Process sequences every 10 seconds
                
            except Exception as e:
                print(f"Error in processing loop: {str(e)}")
                time.sleep(5)  # Short delay on error
    
    def add_alert_callback(self, callback: Callable):
        """Add a callback for alert notifications"""
        self.alert_manager.add_alert_callback(callback)
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'processing_status': {
                'enabled': self.processing_enabled,
                'uptime_hours': self.performance_monitor.get_performance_metrics()['uptime_hours']
            },
            'buffer_status': self.flow_buffer.get_buffer_stats(),
            'processing_stats': self.stats.copy(),
            'performance_metrics': self.performance_monitor.get_performance_metrics(),
            'alert_statistics': self.alert_manager.get_alert_statistics(),
            'recent_alerts': len(self.alert_manager.get_recent_alerts(hours=1))
        }
    
    def get_recent_alerts(self, hours: int = 1) -> List[Dict]:
        """Get recent security alerts"""
        return self.alert_manager.get_recent_alerts(hours)
    
    def export_logs(self, filepath: str):
        """Export processing logs and alerts to file"""
        logs = {
            'timestamp': datetime.now().isoformat(),
            'system_status': self.get_system_status(),
            'recent_alerts': self.get_recent_alerts(hours=24),
            'processing_statistics': self.stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"Logs exported to {filepath}")


def create_streaming_processor(models_dir: str = "models_advanced") -> StreamingProcessor:
    """Factory function to create a streaming processor"""
    try:
        processor = StreamingProcessor(models_dir)
        
        # Add default alert callback (print to console)
        def console_alert_callback(alert):
            level = alert['level']
            timestamp = alert['timestamp']
            pred_class = alert['predicted_class']
            confidence = alert['confidence']
            reasons = ', '.join(alert['alert_reasons'])
            
            print(f"[{level}] {timestamp}: Attack Type {pred_class} "
                  f"(confidence: {confidence:.3f}) - {reasons}")
        
        processor.add_alert_callback(console_alert_callback)
        
        return processor
        
    except Exception as e:
        print(f"Error creating streaming processor: {str(e)}")
        raise


if __name__ == "__main__":
    # Test the streaming processor
    print("Testing Streaming Processor...")
    
    try:
        # Create streaming processor
        processor = create_streaming_processor()
        
        # Start processing
        processor.start_processing()
        
        # Simulate network flows
        print("Simulating network flows...")
        for i in range(50):
            # Create random flow data
            flow_data = np.random.random(41)
            
            # Add some anomalous patterns occasionally
            if i % 10 == 0:
                flow_data += np.random.normal(2, 0.5, 41)  # Anomalous pattern
            
            # Process the flow
            result = processor.process_flow(flow_data)
            
            if i % 10 == 0:
                print(f"Flow {i}: {result.final_prediction} "
                      f"(conf: {result.final_confidence:.3f}, "
                      f"anomaly: {result.is_anomaly})")
            
            time.sleep(0.1)  # Simulate real-time arrival
        
        # Wait a bit for sequence processing
        time.sleep(2)
        
        # Get system status
        status = processor.get_system_status()
        print(f"\nSystem Status:")
        print(f"  Flows processed: {status['processing_stats']['flows_processed']}")
        print(f"  Normal flows: {status['processing_stats']['normal_flows']}")
        print(f"  Attack flows: {status['processing_stats']['attack_flows']}")
        print(f"  Anomalous flows: {status['processing_stats']['anomalous_flows']}")
        print(f"  Recent alerts: {status['recent_alerts']}")
        
        # Stop processing
        processor.stop_processing()
        
        print("Streaming processor test completed!")
        
    except Exception as e:
        print(f"Error testing streaming processor: {str(e)}")
        print("This is expected if the hybrid system hasn't been fully trained yet.")