"""
Neural Predictive Energy Model for Energy Forecasting in MAHIA-X.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from typing import Dict, Any, Tuple, List, Optional
import math
from collections import deque

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from modell_V5_MAHIA_HyenaMoE import GPUTelemetryMonitor
    ENERGY_AVAILABLE = True
except ImportError:
    ENERGY_AVAILABLE = False
    print("⚠️  MAHIA-X modules not available for energy forecasting")


class NeuralEnergyPredictor(nn.Module):
    """Neural network for predicting energy consumption based on system metrics"""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 1,
                 sequence_length: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        Args:
            x: Input tensor of shape (B, T, D) where T is sequence length
        Returns:
            Predicted energy consumption of shape (B, output_dim)
        """
        B, T, D = x.shape
        
        # Normalize features
        x = self.feature_norm(x)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)  # (B, T, H)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (B, T, H)
        
        # Global average pooling over time
        pooled = torch.mean(attn_out, dim=1)  # (B, H)
        
        # Prediction
        prediction = self.predictor(pooled)  # (B, output_dim)
        
        return prediction


class EnergyFeatureExtractor:
    """Extract relevant features for energy prediction"""
    
    def __init__(self):
        self.feature_names = [
            'gpu_utilization',
            'gpu_memory_used',
            'gpu_power_usage',
            'gpu_temperature',
            'cpu_utilization',
            'memory_usage',
            'batch_size',
            'sequence_length',
            'model_complexity',
            'training_step'
        ]
        
    def extract_features(self, telemetry_data: Dict[str, Any], 
                        training_context: Dict[str, Any]) -> np.ndarray:
        """Extract features from telemetry and training context
        Args:
            telemetry_data: GPU/CPU telemetry data
            training_context: Training context information
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # GPU metrics
        features.append(telemetry_data.get('gpu_utilization', 0.0) / 100.0)  # Normalize to [0,1]
        features.append(telemetry_data.get('gpu_memory_used', 0.0) / telemetry_data.get('gpu_memory_total', 1.0))
        features.append(telemetry_data.get('gpu_power_usage', 0.0) / telemetry_data.get('gpu_power_limit', 250.0))
        features.append(telemetry_data.get('gpu_temperature', 0.0) / 100.0)  # Normalize to [0,1]
        
        # CPU and system metrics (simplified)
        features.append(training_context.get('cpu_utilization', 0.0) / 100.0)
        features.append(training_context.get('memory_usage', 0.0) / training_context.get('memory_total', 1.0))
        
        # Training context
        features.append(training_context.get('batch_size', 32) / 1024.0)  # Normalize
        features.append(training_context.get('sequence_length', 64) / 512.0)  # Normalize
        features.append(training_context.get('model_complexity', 1.0))  # Relative complexity
        features.append(training_context.get('training_step', 0) / 10000.0)  # Normalize
        
        return np.array(features, dtype=np.float32)


class NeuralPredictiveEnergyModel:
    """Complete neural predictive energy model for forecasting energy consumption"""
    
    def __init__(self, sequence_length: int = 10, prediction_horizon: int = 5):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Feature extractor
        self.feature_extractor = EnergyFeatureExtractor()
        
        # Neural predictor
        self.predictor = NeuralEnergyPredictor(
            input_dim=len(self.feature_extractor.feature_names),
            hidden_dim=64,
            output_dim=1,
            sequence_length=sequence_length
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)
        
        # Historical data
        self.feature_history = deque(maxlen=1000)
        self.energy_history = deque(maxlen=1000)
        
        # Performance tracking
        self.training_losses = deque(maxlen=100)
        self.prediction_errors = deque(maxlen=100)
        
    def add_observation(self, telemetry_data: Dict[str, Any], 
                       training_context: Dict[str, Any], 
                       actual_energy: float):
        """Add new observation to history
        Args:
            telemetry_data: Current telemetry data
            training_context: Current training context
            actual_energy: Actual energy consumption measured
        """
        # Extract features
        features = self.feature_extractor.extract_features(telemetry_data, training_context)
        
        # Store in history
        self.feature_history.append(features)
        self.energy_history.append(actual_energy)
        
    def prepare_sequences(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare sequences for training
        Returns:
            Tuple of (features_seq, energy_targets)
        """
        if len(self.feature_history) < self.sequence_length + 1:
            return None, None
            
        # Convert to tensors
        features_array = np.array(self.feature_history)
        energy_array = np.array(self.energy_history)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(features_array) - self.sequence_length):
            seq = features_array[i:i+self.sequence_length]
            target = energy_array[i+self.sequence_length]
            sequences.append(seq)
            targets.append(target)
            
        if not sequences:
            return None, None
            
        # Convert to tensors
        features_seq = torch.tensor(np.array(sequences), dtype=torch.float32)
        energy_targets = torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(-1)
        
        return features_seq, energy_targets
    
    def train_step(self) -> Optional[float]:
        """Perform one training step
        Returns:
            Training loss or None if not enough data
        """
        # Prepare sequences
        features_seq, energy_targets = self.prepare_sequences()
        
        if features_seq is None or energy_targets is None:
            return None
            
        # Forward pass
        predictions = self.predictor(features_seq)
        
        # Compute loss
        loss = F.mse_loss(predictions, energy_targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.optimizer.step()
        
        # Store loss
        self.training_losses.append(loss.item())
        
        return loss.item()
    
    def predict_energy(self, future_steps: int = 1) -> List[float]:
        """Predict energy consumption for future steps
        Args:
            future_steps: Number of future steps to predict
        Returns:
            List of predicted energy values
        """
        if len(self.feature_history) < self.sequence_length:
            # Not enough history, return average of recent energy
            if self.energy_history:
                return [np.mean(self.energy_history)] * future_steps
            else:
                return [0.0] * future_steps
                
        # Get recent sequence
        recent_features = list(self.feature_history)[-self.sequence_length:]
        features_seq = torch.tensor(np.array(recent_features), dtype=torch.float32).unsqueeze(0)
        
        predictions = []
        current_seq = features_seq.clone()
        
        # Predict for each future step
        for _ in range(future_steps):
            with torch.no_grad():
                pred = self.predictor(current_seq)
                predictions.append(pred.item())
                
            # For multi-step prediction, we would need to update the sequence
            # This is a simplified approach - in practice, you might use a more sophisticated method
            break
            
        # For remaining steps, use the last prediction
        if len(predictions) < future_steps:
            last_pred = predictions[-1] if predictions else 0.0
            predictions.extend([last_pred] * (future_steps - len(predictions)))
            
        return predictions
    
    def evaluate_prediction(self, predicted: float, actual: float) -> float:
        """Evaluate prediction accuracy
        Args:
            predicted: Predicted energy value
            actual: Actual energy value
        Returns:
            Absolute error
        """
        error = abs(predicted - actual)
        self.prediction_errors.append(error)
        return error
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics
        Returns:
            Dictionary of model statistics
        """
        return {
            'total_observations': len(self.feature_history),
            'avg_training_loss': np.mean(self.training_losses) if self.training_losses else 0.0,
            'avg_prediction_error': np.mean(self.prediction_errors) if self.prediction_errors else 0.0,
            'feature_dimension': len(self.feature_extractor.feature_names),
            'sequence_length': self.sequence_length
        }


class EnergyOptimizer:
    """Energy optimizer that uses energy predictions to optimize training"""
    
    def __init__(self, energy_model: NeuralPredictiveEnergyModel):
        self.energy_model = energy_model
        self.energy_budget = 1000.0  # Joules
        self.current_energy = 0.0
        self.optimization_history = []
        
    def optimize_batch_size(self, current_batch_size: int, 
                           predicted_energy_per_step: float,
                           max_batch_size: int = 128) -> int:
        """Optimize batch size based on energy predictions
        Args:
            current_batch_size: Current batch size
            predicted_energy_per_step: Predicted energy per training step
            max_batch_size: Maximum allowed batch size
        Returns:
            Optimized batch size
        """
        # Calculate remaining energy budget
        remaining_energy = self.energy_budget - self.current_energy
        
        # Estimate steps we can afford
        if predicted_energy_per_step > 0:
            affordable_steps = remaining_energy / predicted_energy_per_step
        else:
            affordable_steps = 1000  # Assume we can afford many steps
            
        # Adjust batch size based on energy constraints
        if affordable_steps < 10:  # If we're running low on energy
            # Reduce batch size to conserve energy
            new_batch_size = max(1, int(current_batch_size * 0.8))
        elif affordable_steps > 100:  # If we have plenty of energy
            # Increase batch size for better throughput
            new_batch_size = min(max_batch_size, int(current_batch_size * 1.1))
        else:
            # Keep current batch size
            new_batch_size = current_batch_size
            
        # Store optimization decision
        self.optimization_history.append({
            'current_batch_size': current_batch_size,
            'new_batch_size': new_batch_size,
            'predicted_energy': predicted_energy_per_step,
            'affordable_steps': affordable_steps
        })
        
        return new_batch_size
    
    def update_energy_consumption(self, energy_consumed: float):
        """Update current energy consumption
        Args:
            energy_consumed: Energy consumed in last step
        """
        self.current_energy += energy_consumed
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics
        Returns:
            Dictionary of optimization statistics
        """
        if not self.optimization_history:
            return {}
            
        recent_optimizations = self.optimization_history[-10:]  # Last 10 optimizations
        batch_size_changes = [opt['new_batch_size'] - opt['current_batch_size'] 
                             for opt in recent_optimizations]
        
        return {
            'total_optimizations': len(self.optimization_history),
            'avg_batch_size_change': np.mean(batch_size_changes),
            'current_energy': self.current_energy,
            'energy_budget': self.energy_budget,
            'energy_remaining': self.energy_budget - self.current_energy
        }


def demo_energy_forecasting():
    """Demonstrate neural predictive energy model"""
    if not ENERGY_AVAILABLE:
        print("❌ MAHIA-X modules not available for energy forecasting")
        return
        
    print("🚀 Demonstrating Neural Predictive Energy Model...")
    print("=" * 60)
    
    # Create energy model
    energy_model = NeuralPredictiveEnergyModel(sequence_length=5, prediction_horizon=3)
    print("✅ Initialized Neural Predictive Energy Model")
    
    # Create energy optimizer
    energy_optimizer = EnergyOptimizer(energy_model)
    print("✅ Initialized Energy Optimizer")
    
    # Simulate telemetry data and training context
    sample_telemetry = {
        'gpu_utilization': 75.0,
        'gpu_memory_used': 4096.0,
        'gpu_memory_total': 8192.0,
        'gpu_power_usage': 150.0,
        'gpu_power_limit': 250.0,
        'gpu_temperature': 65.0
    }
    
    sample_context = {
        'cpu_utilization': 45.0,
        'memory_usage': 8192.0,
        'memory_total': 16384.0,
        'batch_size': 32,
        'sequence_length': 64,
        'model_complexity': 1.0,
        'training_step': 1000
    }
    
    # Add sample observations
    print("✅ Adding sample observations...")
    for i in range(15):
        # Simulate varying energy consumption
        actual_energy = 50.0 + np.random.randn() * 10.0 + i * 0.5  # Increasing trend
        
        # Add observation
        energy_model.add_observation(sample_telemetry, sample_context, actual_energy)
        
        # Update context for next step
        sample_context['training_step'] += 1
        sample_context['batch_size'] = max(8, sample_context['batch_size'] + np.random.randint(-2, 3))
        
    print(f"   Added {len(energy_model.feature_history)} observations")
    
    # Train model
    print("✅ Training energy predictor...")
    for epoch in range(20):
        loss = energy_model.train_step()
        if loss is not None:
            if epoch % 5 == 0:
                print(f"   Epoch {epoch}: Loss = {loss:.4f}")
    
    # Make predictions
    predictions = energy_model.predict_energy(future_steps=3)
    print(f"✅ Energy predictions for next 3 steps: {[f'{p:.2f}J' for p in predictions]}")
    
    # Evaluate prediction accuracy
    if energy_model.energy_history:
        actual = energy_model.energy_history[-1]
        predicted = predictions[0] if predictions else 0.0
        error = energy_model.evaluate_prediction(predicted, actual)
        print(f"✅ Prediction error: {error:.2f}J (Actual: {actual:.2f}J, Predicted: {predicted:.2f}J)")
    
    # Test energy optimization
    print("✅ Testing energy optimization...")
    current_batch_size = 32
    predicted_energy = predictions[0] if predictions else 50.0
    optimized_batch_size = energy_optimizer.optimize_batch_size(
        current_batch_size, predicted_energy, max_batch_size=64
    )
    print(f"   Current batch size: {current_batch_size}")
    print(f"   Optimized batch size: {optimized_batch_size}")
    
    # Update energy consumption
    energy_optimizer.update_energy_consumption(predicted_energy)
    
    # Print model statistics
    model_stats = energy_model.get_model_stats()
    print(f"✅ Model statistics:")
    for key, value in model_stats.items():
        print(f"   {key}: {value}")
        
    # Print optimization statistics
    opt_stats = energy_optimizer.get_optimization_stats()
    print(f"✅ Optimization statistics:")
    for key, value in opt_stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("NEURAL PREDICTIVE ENERGY MODEL DEMO SUMMARY")
    print("=" * 60)
    print("Key Features Implemented:")
    print("  1. LSTM-based energy prediction with attention")
    print("  2. Multi-feature energy modeling")
    print("  3. Online learning and adaptation")
    print("  4. Energy-aware optimization")
    print("  5. Performance monitoring and evaluation")
    print("\nBenefits:")
    print("  - Accurate energy forecasting")
    print("  - Energy-efficient training optimization")
    print("  - Dynamic resource allocation")
    print("  - Carbon footprint reduction")
    
    print("\n✅ Neural Predictive Energy Model demonstration completed!")


def main():
    """Main demonstration function"""
    demo_energy_forecasting()


if __name__ == '__main__':
    main()