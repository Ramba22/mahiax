#!/usr/bin/env python3
"""
Unified Controller Interface for MAHIA-X
This module provides a consolidated interface for all training controllers including:
- ExtendStop: Early stopping with extension capabilities
- PredictiveStopForecaster: Predictive training saturation detection
- MetaLRPolicyController: Meta-learning rate policy controller
- ConfidenceTrendBasedLRAdjuster: Confidence-based learning rate adjustment
"""

# Conditional imports with fallbacks
TORCH_AVAILABLE = False
torch = None
nn = None

NUMPY_AVAILABLE = False
np = None

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass

import time
from typing import Optional, Dict, Any, Tuple, List
import os
import math


class BaseController:
    """Base class for all controllers"""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.history = []
        
    def enable(self):
        """Enable the controller"""
        self.enabled = True
        
    def disable(self):
        """Disable the controller"""
        self.enabled = False
        
    def is_enabled(self) -> bool:
        """Check if controller is enabled"""
        return self.enabled
        
    def get_name(self) -> str:
        """Get controller name"""
        return self.name
        
    def get_history(self) -> List:
        """Get controller history"""
        return self.history.copy()
        
    def reset(self):
        """Reset controller state"""
        self.history = []


class ExtendStopController(BaseController):
    """Enhanced ExtendStop controller v2 with ensemble stopping capabilities"""
    
    def __init__(self, patience: int = 20, min_delta: float = 5e-3, 
                 extend_patience: int = 5, max_extensions: int = 1,
                 entropy_window: int = 10, confidence_window: int = 10, 
                 curvature_window: int = 5):
        super().__init__("ExtendStopV2")
        self.patience = patience
        self.min_delta = min_delta
        self.extend_patience = extend_patience
        self.max_extensions = max_extensions
        
        self.best_loss = float('inf')
        self.counter = 0
        self.extensions_used = 0
        self.stop_training = False
        self.loss_history = []
        self.metric_history = []
        self.checkpoint_saved = False
        
        # Ensemble components
        self.entropy_window = entropy_window
        self.confidence_window = confidence_window
        self.curvature_window = curvature_window
        self.entropy_trend_history = []
        self.confidence_variance_history = []
        self.loss_curvature_history = []
        
        # Soft-pause functionality
        self.soft_pause_active = False
        self.gradient_flow_history = []
        self.silent_layer_count = 0
        self.total_layers = 0
        
        # Integration with PredictiveStopForecaster
        self.predictive_forecaster = None
        self.use_predictive_forecasting = True
        
        # RNN/SSM prediction for stagnation detection
        self.stagnation_predictor = None
        self.prediction_window = 5  # Predict 3-5 batches ahead
        
    def set_predictive_forecaster(self, forecaster: 'PredictiveStopController'):
        """Link to predictive forecaster"""
        self.predictive_forecaster = forecaster
        
    def _compute_entropy_trend(self, loss_history: list) -> float:
        """Compute entropy trend from loss history"""
        if len(loss_history) < self.entropy_window:
            return 0.0
            
        # Get recent window of losses
        recent_losses = loss_history[-self.entropy_window:]
        
        # Convert to probability distribution
        if NUMPY_AVAILABLE and np is not None:
            losses_array = np.array(recent_losses)
            # Normalize to probability distribution
            if np.sum(losses_array) > 0:
                prob_dist = losses_array / np.sum(losses_array)
                # Compute entropy
                entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-12))
                return float(entropy)
        else:
            # Fallback implementation without numpy
            total = sum(recent_losses)
            if total > 0:
                prob_dist = [loss / total for loss in recent_losses]
                entropy = -sum(p * math.log(p + 1e-12) for p in prob_dist if p > 0)
                return entropy
        return 0.0
        
    def _compute_confidence_variance(self, metric_history: list) -> float:
        """Compute variance in validation metric as confidence measure"""
        if len(metric_history) < self.confidence_window:
            return 0.0
            
        # Get recent window of metrics
        recent_metrics = metric_history[-self.confidence_window:]
        
        if NUMPY_AVAILABLE and np is not None:
            return float(np.var(recent_metrics))
        else:
            # Fallback implementation without numpy
            mean = sum(recent_metrics) / len(recent_metrics)
            variance = sum((x - mean) ** 2 for x in recent_metrics) / len(recent_metrics)
            return variance
            
    def _compute_loss_curvature(self, loss_history: list) -> float:
        """Compute curvature of loss curve as second derivative"""
        if len(loss_history) < self.curvature_window:
            return 0.0
            
        # Get recent window of losses
        recent_losses = loss_history[-self.curvature_window:]
        
        if len(recent_losses) < 3:
            return 0.0
            
        # Simple second derivative approximation
        if NUMPY_AVAILABLE and np is not None:
            losses_array = np.array(recent_losses)
            # Second derivative using central difference
            if len(losses_array) >= 3:
                second_deriv = losses_array[:-2] - 2 * losses_array[1:-1] + losses_array[2:]
                return float(np.mean(second_deriv))
        else:
            # Fallback implementation without numpy
            second_derivs = []
            for i in range(1, len(recent_losses) - 1):
                second_deriv = recent_losses[i-1] - 2 * recent_losses[i] + recent_losses[i+1]
                second_derivs.append(second_deriv)
            if second_derivs:
                return sum(second_derivs) / len(second_derivs)
        return 0.0
        
    def _check_ensemble_stopping(self, current_loss: float, current_metric: Optional[float]) -> dict:
        """Check ensemble stopping criteria"""
        # Update histories
        self.loss_history.append(current_loss)
        if current_metric is not None:
            self.metric_history.append(current_metric)
        
        # Compute ensemble components
        entropy_trend = self._compute_entropy_trend(self.loss_history)
        confidence_variance = self._compute_confidence_variance(self.metric_history) if self.metric_history else 0.0
        loss_curvature = self._compute_loss_curvature(self.loss_history)
        
        # Store for history tracking
        self.entropy_trend_history.append(entropy_trend)
        self.confidence_variance_history.append(confidence_variance)
        self.loss_curvature_history.append(loss_curvature)
        
        # Ensemble decision logic
        ensemble_stop = False
        reasons = []
        
        # Entropy trend indicates stagnation
        if len(self.entropy_trend_history) >= 3:
            recent_entropy = self.entropy_trend_history[-3:]
            # If entropy is very low, training may have stagnated
            if all(e < 0.01 for e in recent_entropy) and len(recent_entropy) >= 3:
                ensemble_stop = True
                reasons.append("low_entropy_trend")
                
        # Confidence variance indicates instability
        if len(self.confidence_variance_history) >= 3:
            recent_variance = self.confidence_variance_history[-3:]
            # If variance is very low, model may not be learning
            if all(v < 1e-6 for v in recent_variance) and len(recent_variance) >= 3:
                ensemble_stop = True
                reasons.append("low_confidence_variance")
                
        # Loss curvature indicates plateau
        if len(self.loss_curvature_history) >= 3:
            recent_curvature = self.loss_curvature_history[-3:]
            # If curvature is near zero, loss may have plateaued
            if all(abs(c) < 1e-6 for c in recent_curvature) and len(recent_curvature) >= 3:
                ensemble_stop = True
                reasons.append("loss_plateau_detected")
                
        return {
            "ensemble_stop": ensemble_stop,
            "reasons": reasons,
            "entropy_trend": entropy_trend,
            "confidence_variance": confidence_variance,
            "loss_curvature": loss_curvature
        }
        
    def _check_gradient_flow(self, model) -> dict:
        """Check gradient flow for soft-pause functionality"""
        if not TORCH_AVAILABLE or not hasattr(model, 'parameters') or torch is None:
            return {"gradient_flow_ok": True, "silent_layers": 0, "total_layers": 0}
            
        silent_count = 0
        total_count = 0
        
        # Check gradients for all parameters
        for param in model.parameters():
            if param.grad is not None:
                total_count += 1
                # Check if gradient is effectively zero (silent layer)
                grad_norm = torch.norm(param.grad) if torch is not None else 0
                if grad_norm < 1e-8:
                    silent_count += 1
                    
        silent_ratio = silent_count / max(total_count, 1)
        
        # Check if too many layers are silent
        too_many_silent = silent_ratio > 0.2  # >20% silent layers
        
        return {
            "gradient_flow_ok": not too_many_silent,
            "silent_layers": silent_count,
            "total_layers": total_count,
            "silent_ratio": silent_ratio,
            "too_many_silent": too_many_silent
        }
        
    def _soft_pause_check(self, model, current_loss: float) -> dict:
        """Perform soft-pause check with gradient flow analysis"""
        gradient_info = self._check_gradient_flow(model)
        
        # Store gradient flow info
        self.gradient_flow_history.append(gradient_info)
        self.silent_layer_count = gradient_info["silent_layers"]
        self.total_layers = gradient_info["total_layers"]
        
        # Warn if too many silent layers
        if gradient_info["too_many_silent"]:
            print(f"⚠️  ExtendStopV2: {gradient_info['silent_ratio']*100:.1f}% layers are silent "
                  f"({gradient_info['silent_layers']}/{gradient_info['total_layers']})")
        
        # If gradient flow is problematic, consider soft pause
        if not gradient_info["gradient_flow_ok"]:
            self.soft_pause_active = True
            return {
                "action": "soft_pause",
                "gradient_flow_ok": False,
                "silent_layers": gradient_info["silent_layers"],
                "total_layers": gradient_info["total_layers"]
            }
        else:
            self.soft_pause_active = False
            return {
                "action": "continue",
                "gradient_flow_ok": True
            }
            
    def __call__(self, current_loss: float, current_metric: Optional[float] = None, 
                 current_epoch: Optional[int] = None, model = None) -> Dict[str, Any]:
        """Returns dict with action recommendations"""
        # Perform ensemble stopping check
        ensemble_result = self._check_ensemble_stopping(current_loss, current_metric)
        
        # Perform soft-pause check if model is provided
        soft_pause_result = None
        if model is not None:
            soft_pause_result = self._soft_pause_check(model, current_loss)
            
        # Traditional loss improvement check
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            self.checkpoint_saved = False
            action = "improving"
        else:
            self.counter += 1
            action = "waiting"
            
        # Use predictive forecasting if available
        predictive_result = None
        if (self.use_predictive_forecasting and self.predictive_forecaster and 
            current_metric is not None and current_epoch is not None):
            try:
                predictive_result = self.predictive_forecaster.predict_saturation(
                    current_loss, current_metric, current_epoch)
            except Exception as e:
                print(f"⚠️  Predictive forecasting failed: {e}")
                predictive_result = None
        
        # Check if we should stop based on predictive forecasting
        if (predictive_result and predictive_result["saturation_predicted"] and 
            predictive_result["confidence"] > 0.7):
            # Use predictive result to make early decision
            if (predictive_result["epochs_until_saturation"] is not None and 
                predictive_result["epochs_until_saturation"] <= 2):
                self.stop_training = True
                print(f"⏹️ {self.name}: Stopping training based on predictive forecast (saturation in {predictive_result['epochs_until_saturation']} epochs)")
                return {"action": "stop", "reduce_lr": True, "save_checkpoint": not self.checkpoint_saved, "predictive_stop": True}
        
        # Check if we should stop based on ensemble criteria
        if ensemble_result["ensemble_stop"]:
            self.stop_training = True
            reasons_str = ", ".join(ensemble_result["reasons"])
            print(f"⏹️ {self.name}: Stopping training based on ensemble criteria ({reasons_str})")
            return {"action": "stop", "reduce_lr": True, "save_checkpoint": not self.checkpoint_saved, "ensemble_stop": True, "reasons": ensemble_result["reasons"]}
        
        # Handle soft pause
        if soft_pause_result and soft_pause_result["action"] == "soft_pause":
            print(f"⏸️ {self.name}: Soft-pause activated - checking gradient flow")
            return {"action": "soft_pause", "reduce_lr": False, "save_checkpoint": not self.checkpoint_saved, "gradient_check": True}
        
        # Check if we should stop based on traditional criteria
        if self.counter >= self.patience:
            if self.extensions_used < self.max_extensions:
                # Extend training
                self.extensions_used += 1
                self.counter = 0
                self.patience += self.extend_patience
                print(f"🔄 {self.name}: Extending training (extension {self.extensions_used}/{self.max_extensions})")
                return {"action": "extend", "reduce_lr": True, "save_checkpoint": not self.checkpoint_saved, "predictive_stop": False}
            else:
                # Stop training
                self.stop_training = True
                print(f"⏹️ {self.name}: Stopping training after {self.extensions_used} extensions")
                return {"action": "stop", "reduce_lr": True, "save_checkpoint": not self.checkpoint_saved, "predictive_stop": False}
        elif self.counter >= self.patience - 2 and not self.checkpoint_saved:
            # Save checkpoint before potential stop
            self.checkpoint_saved = True
            return {"action": action, "reduce_lr": False, "save_checkpoint": True, "predictive_stop": False}
        elif self.extensions_used >= self.max_extensions and self.counter >= self.patience:
            # Force stop only after full patience period when all extensions used
            self.stop_training = True
            print(f"⏹️ {self.name}: Forcing stop after {self.extensions_used} extensions")
            return {"action": "stop", "reduce_lr": True, "save_checkpoint": not self.checkpoint_saved, "predictive_stop": False}
                
        return {"action": action, "reduce_lr": False, "save_checkpoint": False, "predictive_stop": False, "ensemble_metrics": ensemble_result}
        
    def should_stop(self) -> bool:
        """Check if training should stop"""
        return self.stop_training
        
    def get_status(self) -> Dict[str, Any]:
        """Get controller status"""
        status = {
            "name": self.name,
            "enabled": self.enabled,
            "best_loss": self.best_loss,
            "counter": self.counter,
            "patience": self.patience,
            "extensions_used": self.extensions_used,
            "max_extensions": self.max_extensions,
            "stop_training": self.stop_training,
            "soft_pause_active": self.soft_pause_active,
            "silent_layer_count": self.silent_layer_count,
            "total_layers": self.total_layers
        }
        
        # Add ensemble metrics if available
        if self.entropy_trend_history:
            status["latest_entropy_trend"] = self.entropy_trend_history[-1]
        if self.confidence_variance_history:
            status["latest_confidence_variance"] = self.confidence_variance_history[-1]
        if self.loss_curvature_history:
            status["latest_loss_curvature"] = self.loss_curvature_history[-1]
            
        return status


class PredictiveStopController(BaseController):
    """Enhanced Predictive Stop Forecaster with RNN/SSM for 3-5 batches ahead stagnation detection"""
    
    def __init__(self, window_size: int = 10, prediction_horizon: int = 5):
        super().__init__("PredictiveStopV2")
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon  # Predict 3-5 batches ahead
        
        # Metrics history for training
        self.loss_history = []
        self.metric_history = []
        self.epoch_history = []
        
        # Simple linear regression parameters (initialized to zero)
        self.slope = 0.0
        self.intercept = 0.0
        self.last_prediction = None
        self.epochs_until_saturation = None
        
        # Enhanced RNN-based predictor for multi-step ahead prediction
        self.rnn_predictor = None
        self.use_rnn = False
        self.rnn_horizon = prediction_horizon  # Predict 3-5 steps ahead
        
        # Enhanced SSM-based predictor
        self.ssm_predictor = None
        self.use_ssm = False
        
        # Multi-step prediction results
        self.multi_step_predictions = []
        self.stagnation_detected = False
        self.stagnation_confidence = 0.0
        
    def _compute_trend(self, values: list) -> tuple:
        """Compute linear trend using least squares"""
        if len(values) < 2:
            return 0.0, values[-1] if values else 0.0
            
        if not NUMPY_AVAILABLE:
            # Simple fallback implementation
            n = len(values)
            x = list(range(n))
            y = values
            
            # Compute means
            mean_x = sum(x) / n
            mean_y = sum(y) / n
            
            # Compute slope and intercept using least squares
            numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
            denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
            
            if denominator > 1e-12:
                slope = numerator / denominator
            else:
                slope = 0.0
                
            intercept = mean_y - slope * mean_x
            return slope, intercept
            
        # NumPy implementation - guarded access
        if np is not None:
            n = len(values)
            x = np.arange(n)
            y = np.array(values)
            
            # Compute slope and intercept using least squares
            slope = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 1e-12 else 0.0
            intercept = np.mean(y) - slope * np.mean(x)
            
            return slope, intercept
        else:
            # Fallback if np is None despite NUMPY_AVAILABLE being True
            n = len(values)
            x = list(range(n))
            y = values
            
            # Compute means
            mean_x = sum(x) / n
            mean_y = sum(y) / n
            
            # Compute slope and intercept using least squares
            numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
            denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
            
            if denominator > 1e-12:
                slope = numerator / denominator
            else:
                slope = 0.0
                
            intercept = mean_y - slope * mean_x
            return slope, intercept
        
    def _init_rnn_predictor(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2):
        """Initialize enhanced RNN-based predictor for multi-step prediction"""
        if not TORCH_AVAILABLE:
            print(f"⚠️  {self.name}: PyTorch not available, cannot initialize RNN predictor")
            self.use_rnn = False
            return
            
        try:
            # Enhanced LSTM with more capacity for multi-step prediction
            self.rnn_predictor = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
            # Output layer for multi-step prediction
            self.rnn_output_layer = nn.Linear(hidden_size, self.rnn_horizon)  # Predict multiple steps ahead
            self.use_rnn = True
            print(f"✅ {self.name}: Enhanced RNN-based predictor initialized for {self.rnn_horizon}-step ahead prediction")
        except Exception as e:
            print(f"⚠️  {self.name}: Failed to initialize RNN predictor: {e}")
            self.use_rnn = False
            
    def _init_ssm_predictor(self, state_dim: int = 32, obs_dim: int = 1):
        """Initialize enhanced SSM-based predictor"""
        if not NUMPY_AVAILABLE:
            print(f"⚠️  {self.name}: NumPy not available, cannot initialize SSM predictor")
            self.use_ssm = False
            return
            
        try:
            # Enhanced state-space model parameters with larger state dimension
            self.ssm_state_dim = state_dim
            self.ssm_obs_dim = obs_dim
            
            # Initialize SSM parameters
            if np is not None:
                # State transition matrix with better stability properties
                self.ssm_A = np.random.randn(state_dim, state_dim) * 0.05
                # Ensure spectral radius < 1 for stability
                eigenvals = np.linalg.eigvals(self.ssm_A)
                if np.max(np.abs(eigenvals)) >= 1.0:
                    self.ssm_A = self.ssm_A * 0.9 / np.max(np.abs(eigenvals))
                    
                self.ssm_B = np.random.randn(state_dim, obs_dim) * 0.1    # Control matrix
                self.ssm_C = np.random.randn(obs_dim, state_dim) * 0.1    # Observation matrix
                self.ssm_state = np.zeros((state_dim, 1))                 # Initial state
                
            self.use_ssm = True
            print(f"✅ {self.name}: Enhanced SSM-based predictor initialized")
        except Exception as e:
            print(f"⚠️  {self.name}: Failed to initialize SSM predictor: {e}")
            self.use_ssm = False
            
    def _rnn_predict_multi_step(self, sequence: list, steps_ahead: int = 5) -> list:
        """Make multi-step prediction using RNN model"""
        if not self.use_rnn or self.rnn_predictor is None or not TORCH_AVAILABLE:
            return [0.0] * steps_ahead
            
        try:
            # Convert to tensor
            seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
            
            # Forward pass
            with torch.no_grad():
                output, _ = self.rnn_predictor(seq_tensor)
                # Use the output layer to predict multiple steps ahead
                predictions = self.rnn_output_layer(output[:, -1, :])  # Output multiple values
                
            # Convert to list
            pred_list = predictions.squeeze().tolist()
            if isinstance(pred_list, float):
                pred_list = [pred_list]
                
            # Ensure we have the right number of predictions
            if len(pred_list) < steps_ahead:
                pred_list.extend([pred_list[-1]] * (steps_ahead - len(pred_list)))
            elif len(pred_list) > steps_ahead:
                pred_list = pred_list[:steps_ahead]
                
            return pred_list
        except Exception as e:
            print(f"⚠️  {self.name}: RNN multi-step prediction failed: {e}")
            return [0.0] * steps_ahead
            
    def _ssm_predict_multi_step(self, observations: list, steps_ahead: int = 5) -> list:
        """Make multi-step prediction using SSM model"""
        if not self.use_ssm or not NUMPY_AVAILABLE or np is None:
            return [0.0] * steps_ahead
            
        try:
            predictions = []
            current_state = self.ssm_state.copy()
            
            # Predict multiple steps ahead
            for i in range(steps_ahead):
                if i < len(observations):
                    # Use actual observations when available
                    control_input = np.array([[observations[-(i+1)]]])
                else:
                    # Use predicted observations when actual not available
                    control_input = np.array([[predictions[-1] if predictions else observations[-1]]])
                    
                # Update state: x_{t+1} = A * x_t + B * u_t
                current_state = np.dot(self.ssm_A, current_state) + np.dot(self.ssm_B, control_input)
                
                # Predict observation: y_{t+1} = C * x_{t+1}
                prediction = np.dot(self.ssm_C, current_state)
                predictions.append(prediction.item())
                
            return predictions
        except Exception as e:
            print(f"⚠️  {self.name}: SSM multi-step prediction failed: {e}")
            return [0.0] * steps_ahead
            
    def _detect_stagnation(self, predictions: list) -> tuple:
        """Detect stagnation in multi-step predictions"""
        if len(predictions) < 3:
            return False, 0.0
            
        # Check if predictions are converging (stagnation indicator)
        differences = [abs(predictions[i] - predictions[i-1]) for i in range(1, len(predictions))]
        mean_diff = sum(differences) / len(differences)
        
        # If differences are very small, it indicates stagnation
        stagnation_threshold = 1e-4
        is_stagnating = mean_diff < stagnation_threshold
        
        # Confidence based on how small the differences are
        confidence = max(0.0, 1.0 - (mean_diff / stagnation_threshold))
        
        return is_stagnating, confidence
        
    def predict_saturation(self, current_loss: float, current_metric: float, 
                          current_epoch: int) -> Dict[str, Any]:
        """Enhanced prediction with 3-5 batches ahead stagnation detection
        
        Args:
            current_loss: Current training loss
            current_metric: Current validation metric
            current_epoch: Current training epoch
            
        Returns:
            dict: Enhanced prediction results with multi-step ahead detection
        """
        # Update history
        self.loss_history.append(current_loss)
        self.metric_history.append(current_metric)
        self.epoch_history.append(current_epoch)
        
        # Keep only recent history
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
            self.metric_history.pop(0)
            self.epoch_history.pop(0)
            
        prediction_result = {
            "saturation_predicted": False,
            "epochs_until_saturation": None,
            "confidence": 0.0,
            "trend_slope": 0.0,
            "recommendation": "continue",
            "rnn_prediction": None,
            "ssm_prediction": None,
            "multi_step_predictions": [],
            "stagnation_detected": False,
            "stagnation_confidence": 0.0,
            "steps_ahead_predicted": self.prediction_horizon
        }
        
        # Initialize predictors if needed
        if len(self.loss_history) >= 5 and self.rnn_predictor is None:
            self._init_rnn_predictor()
            self._init_ssm_predictor()
        
        # Enhanced RNN-based multi-step prediction
        if self.use_rnn and len(self.metric_history) >= 5:
            try:
                multi_step_preds = self._rnn_predict_multi_step(
                    self.metric_history[-min(10, len(self.metric_history)):], 
                    self.prediction_horizon
                )
                prediction_result["rnn_prediction"] = multi_step_preds[0]  # First prediction
                prediction_result["multi_step_predictions"] = multi_step_preds
                
                # Detect stagnation in multi-step predictions
                stagnating, confidence = self._detect_stagnation(multi_step_preds)
                prediction_result["stagnation_detected"] = stagnating
                prediction_result["stagnation_confidence"] = confidence
                
                # Store for history tracking
                self.multi_step_predictions = multi_step_preds
                self.stagnation_detected = stagnating
                self.stagnation_confidence = confidence
                
                # If stagnation detected with high confidence, recommend stopping
                if stagnating and confidence > 0.7:
                    prediction_result["saturation_predicted"] = True
                    prediction_result["epochs_until_saturation"] = 1  # Imminent saturation
                    prediction_result["confidence"] = confidence
                    prediction_result["recommendation"] = "stagnation_detected"
                    self.epochs_until_saturation = 1
                    
            except Exception as e:
                print(f"⚠️  {self.name}: Enhanced RNN prediction error: {e}")
        
        # Enhanced SSM-based prediction
        if self.use_ssm and len(self.metric_history) >= 1:
            try:
                # Use both RNN and SSM predictions for ensemble
                ssm_multi_preds = self._ssm_predict_multi_step(
                    self.metric_history[-min(5, len(self.metric_history)):], 
                    self.prediction_horizon
                )
                prediction_result["ssm_prediction"] = ssm_multi_preds[0]  # First prediction
                
                # If we don't have RNN predictions, use SSM for stagnation detection
                if not prediction_result["stagnation_detected"] and prediction_result["ssm_prediction"]:
                    stagnating, confidence = self._detect_stagnation(ssm_multi_preds)
                    # Combine with existing confidence if available
                    if prediction_result["stagnation_confidence"] > 0:
                        combined_confidence = (prediction_result["stagnation_confidence"] + confidence) / 2
                    else:
                        combined_confidence = confidence
                        
                    prediction_result["stagnation_detected"] = stagnating or prediction_result["stagnation_detected"]
                    prediction_result["stagnation_confidence"] = combined_confidence
                    
            except Exception as e:
                print(f"⚠️  {self.name}: Enhanced SSM prediction error: {e}")
        
        # Traditional trend-based prediction as fallback
        if len(self.loss_history) >= 5:
            # Compute trend in validation metric
            slope, intercept = self._compute_trend(self.metric_history)
            self.slope = slope
            self.intercept = intercept
            
            prediction_result["trend_slope"] = slope
            
            # If slope is near zero, we might be approaching saturation
            if abs(slope) < 1e-4:
                prediction_result["saturation_predicted"] = True
                prediction_result["epochs_until_saturation"] = 0
                prediction_result["confidence"] = 0.9
                prediction_result["recommendation"] = "saturation_detected"
                self.epochs_until_saturation = 0
            elif slope < 0:  # Metric is decreasing (getting worse)
                # Predict when it will stop decreasing
                prediction_result["saturation_predicted"] = True
                prediction_result["epochs_until_saturation"] = max(1, int(abs(slope) * 10))
                prediction_result["confidence"] = 0.7
                prediction_result["recommendation"] = "potential_saturation"
                self.epochs_until_saturation = prediction_result["epochs_until_saturation"]
            else:  # Metric is improving
                # Predict when improvement will slow down significantly
                improvement_rate = slope
                if improvement_rate < 0.001:  # Very slow improvement
                    prediction_result["saturation_predicted"] = True
                    prediction_result["epochs_until_saturation"] = max(1, int(0.01 / max(improvement_rate, 1e-8)))
                    prediction_result["confidence"] = 0.6
                    prediction_result["recommendation"] = "slow_improvement"
                    self.epochs_until_saturation = prediction_result["epochs_until_saturation"]
                
        self.last_prediction = prediction_result
        return prediction_result
        
    def should_early_stop(self) -> bool:
        """Determine if training should be stopped based on predictions"""
        if self.last_prediction and self.last_prediction["saturation_predicted"]:
            # If we predict saturation within the horizon, consider stopping
            if (self.epochs_until_saturation is not None and 
                self.epochs_until_saturation <= self.prediction_horizon):
                return True
            # Also stop if stagnation is detected with high confidence
            if (self.last_prediction["stagnation_detected"] and 
                self.last_prediction["stagnation_confidence"] > 0.8):
                return True
        return False
        
    def get_status(self) -> Dict[str, Any]:
        """Get controller status"""
        status = {
            "name": self.name,
            "enabled": self.enabled,
            "window_size": self.window_size,
            "prediction_horizon": self.prediction_horizon,
            "loss_history_length": len(self.loss_history),
            "metric_history_length": len(self.metric_history),
            "use_rnn": self.use_rnn,
            "use_ssm": self.use_ssm,
            "last_slope": self.slope,
            "last_intercept": self.intercept
        }
        
        # Add enhanced metrics
        if self.multi_step_predictions:
            status["latest_multi_step_predictions"] = self.multi_step_predictions
        status["stagnation_detected"] = self.stagnation_detected
        status["stagnation_confidence"] = self.stagnation_confidence
            
        return status


class MetaLRPolicyController(BaseController):
    """Unified Meta-learning rate policy controller using reinforcement learning principles"""
    
    def __init__(self, state_dim: int = 8, action_dim: int = 3, lr: float = 1e-3):
        super().__init__("MetaLRPolicy")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        
        # Policy network (simplified implementation)
        self.policy_network = None
        self.value_network = None
        self.use_rl = False
        
        # Initialize policy if PyTorch is available
        self._init_policy_network()
        
        # State history
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
        # Performance tracking
        self.performance_window = []
        self.window_size = 10
        
    def _init_policy_network(self):
        """Initialize policy network for RL-based LR control"""
        if not TORCH_AVAILABLE:
            print(f"⚠️  {self.name}: PyTorch not available, cannot initialize RL policy")
            self.use_rl = False
            return
            
        try:
            # Policy network: state -> action probabilities
            self.policy_network = nn.Sequential(
                nn.Linear(self.state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, self.action_dim),
                nn.Softmax(dim=-1)
            )
            
            # Value network: state -> value
            self.value_network = nn.Sequential(
                nn.Linear(self.state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            
            self.use_rl = True
            print(f"✅ {self.name}: Initialized with RL")
        except Exception as e:
            print(f"⚠️  {self.name}: Failed to initialize RL policy controller: {e}")
            self.use_rl = False
            
    def _get_state(self, current_loss: float, loss_improvement: float, 
                   current_lr: float, step: int) -> list:
        """Extract state representation from training metrics"""
        # Simple state representation
        if NUMPY_AVAILABLE and np is not None:
            state = np.array([
                current_loss,
                loss_improvement,
                current_lr,
                step,
                np.mean(self.performance_window) if self.performance_window else 0.0,
                np.std(self.performance_window) if len(self.performance_window) > 1 else 0.0,
                len(self.state_history),
                time.time() % 1000  # Time-based feature
            ], dtype=np.float32)
            
            return state.tolist()
        else:
            # Fallback without NumPy
            mean_perf = sum(self.performance_window) / len(self.performance_window) if self.performance_window else 0.0
            std_perf = 0.0
            if len(self.performance_window) > 1:
                variance = sum((x - mean_perf) ** 2 for x in self.performance_window) / len(self.performance_window)
                std_perf = variance ** 0.5
                
            state = [
                current_loss,
                loss_improvement,
                current_lr,
                step,
                mean_perf,
                std_perf,
                len(self.state_history),
                time.time() % 1000  # Time-based feature
            ]
            
            return state
        
    def _compute_reward(self, loss_improvement: float, lr_change: float) -> float:
        """Compute reward for LR adjustment action"""
        # Reward based on loss improvement
        if loss_improvement > 1e-4:
            reward = 1.0  # Good improvement
        elif loss_improvement > 0:
            reward = 0.1  # Small improvement
        elif loss_improvement > -1e-4:
            reward = -0.1  # No improvement
        else:
            reward = -1.0  # Loss getting worse
            
        # Penalize excessive LR changes
        if abs(lr_change) > 0.1:
            reward -= 0.5
            
        return reward
        
    def get_lr_action(self, current_loss: float, loss_improvement: float, 
                     current_lr: float, step: int) -> Dict[str, Any]:
        """Get learning rate adjustment action based on current state"""
        # Update performance window
        self.performance_window.append(current_loss)
        if len(self.performance_window) > self.window_size:
            self.performance_window.pop(0)
            
        # Get state representation
        state = self._get_state(current_loss, loss_improvement, current_lr, step)
        
        # Store state
        self.state_history.append(state)
        
        # Use RL policy if available
        if self.use_rl and self.policy_network and TORCH_AVAILABLE:
            try:
                # Convert state to tensor
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                
                # Get action probabilities
                with torch.no_grad():
                    action_probs = self.policy_network(state_tensor)
                    action = torch.multinomial(action_probs, 1).item()
                    
                # Map action to LR adjustment
                if action == 0:  # Reduce LR
                    action_name = "reduce_lr"
                elif action == 1:  # Increase LR
                    action_name = "increase_lr"
                else:  # Maintain LR
                    action_name = "maintain_lr"
                    
                return {
                    "action": action_name,
                    "confidence": action_probs[0, action].item(),
                    "state": state
                }
            except Exception as e:
                print(f"⚠️  {self.name}: RL policy failed: {e}")
                
        # Fallback to rule-based policy
        if loss_improvement < -1e-4:  # Loss getting significantly worse
            action = "reduce_lr"
        elif loss_improvement > 1e-3:  # Good improvement
            action = "increase_lr"
        else:
            action = "maintain_lr"
            
        return {
            "action": action,
            "confidence": 0.8,
            "state": state
        }
        
    def update_policy(self, reward: float):
        """Update policy based on received reward (simplified)"""
        self.reward_history.append(reward)
        
        # Keep only recent history
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
            
        # In a full implementation, this would perform policy gradient updates
        # For now, we just log the reward
        if len(self.reward_history) % 10 == 0:
            if NUMPY_AVAILABLE and np is not None:
                avg_reward = np.mean(self.reward_history[-10:])
            else:
                recent_rewards = self.reward_history[-min(10, len(self.reward_history)):]
                avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
            print(f"📈 {self.name}: Average reward: {avg_reward:.4f}")
            
    def get_policy_stats(self) -> Dict[str, Any]:
        """Get policy statistics"""
        if not self.reward_history:
            return {}
            
        if NUMPY_AVAILABLE and np is not None:
            avg_reward = np.mean(self.reward_history) if self.reward_history else 0.0
            std_reward = np.std(self.reward_history) if len(self.reward_history) > 1 else 0.0
            recent_rewards = self.reward_history[-min(5, len(self.reward_history)):] if self.reward_history else []
        else:
            avg_reward = sum(self.reward_history) / len(self.reward_history) if self.reward_history else 0.0
            variance = sum((x - avg_reward) ** 2 for x in self.reward_history) / len(self.reward_history) if len(self.reward_history) > 1 else 0.0
            std_reward = variance ** 0.5 if len(self.reward_history) > 1 else 0.0
            recent_rewards = self.reward_history[-min(5, len(self.reward_history)):] if self.reward_history else []
            
        return {
            "name": self.name,
            "enabled": self.enabled,
            "total_actions": len(self.action_history),
            "total_rewards": len(self.reward_history),
            "average_reward": avg_reward,
            "reward_std": std_reward,
            "recent_rewards": recent_rewards
        }
        
    def get_status(self) -> Dict[str, Any]:
        """Get controller status"""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "use_rl": self.use_rl,
            "history_length": len(self.state_history)
        }


class ConfidenceTrendBasedLRAdjuster(BaseController):
    """Unified Confidence-trend-based learning rate adjustment"""
    
    def __init__(self, window_size: int = 10, adjustment_factor: float = 0.9,
                 min_lr: float = 1e-7, max_lr: float = 1e-2):
        super().__init__("ConfidenceTrendLR")
        self.window_size = window_size
        self.adjustment_factor = adjustment_factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        
        # Confidence and LR history
        self.confidence_history = []
        self.lr_history = []
        self.trend_counter = 0
        
    def adjust_lr_based_on_confidence_trend(self, optimizer: Any, current_confidence: float, 
                                          current_lr: float) -> Dict[str, Any]:
        """Adjust learning rate based on confidence trends over time
        
        Args:
            optimizer: Training optimizer
            current_confidence: Current model confidence score
            current_lr: Current learning rate
            
        Returns:
            dict: Adjustment results and recommendations
        """
        self.confidence_history.append(current_confidence)
        self.lr_history.append(current_lr)
        
        # Keep only recent history
        if len(self.confidence_history) > self.window_size:
            self.confidence_history.pop(0)
            self.lr_history.pop(0)
            
        adjustment_result = {
            "lr_adjusted": False,
            "new_lr": current_lr,
            "recommendation": "no_change",
            "confidence_trend": 0.0
        }
        
        # Need sufficient history to compute trends
        if len(self.confidence_history) >= 5:
            # Compute confidence trend using linear regression
            if NUMPY_AVAILABLE and np is not None:
                x = np.arange(len(self.confidence_history))
                y = np.array(self.confidence_history)
                
                # Compute slope (trend)
                if np.var(x) > 1e-12:
                    slope = np.cov(x, y)[0, 1] / np.var(x)
                else:
                    slope = 0.0
            else:
                # Fallback without NumPy
                n = len(self.confidence_history)
                if n > 1:
                    x = list(range(n))
                    y = self.confidence_history
                    
                    # Compute means
                    mean_x = sum(x) / n
                    mean_y = sum(y) / n
                    
                    # Compute slope using least squares
                    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
                    denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
                    
                    if denominator > 1e-12:
                        slope = numerator / denominator
                    else:
                        slope = 0.0
                else:
                    slope = 0.0
                
            adjustment_result["confidence_trend"] = slope
            
            # Adjust LR based on confidence trend
            if slope > 0.01:  # Confidence is increasing
                # Increase LR to accelerate learning
                new_lr = min(current_lr / self.adjustment_factor, self.max_lr)
                if new_lr > current_lr:
                    # Note: In a real implementation, we would modify the optimizer's learning rate
                    # For this example, we'll just simulate it
                    adjustment_result["lr_adjusted"] = True
                    adjustment_result["new_lr"] = new_lr
                    adjustment_result["recommendation"] = "increase_lr"
                    print(f"📈 {self.name}: Increasing LR to {new_lr:.2e} (confidence trend: {slope:.4f})")
                    
            elif slope < -0.01:  # Confidence is decreasing
                # Decrease LR to stabilize learning
                new_lr = max(current_lr * self.adjustment_factor, self.min_lr)
                if new_lr < current_lr:
                    # Note: In a real implementation, we would modify the optimizer's learning rate
                    # For this example, we'll just simulate it
                    adjustment_result["lr_adjusted"] = True
                    adjustment_result["new_lr"] = new_lr
                    adjustment_result["recommendation"] = "decrease_lr"
                    print(f"📉 {self.name}: Decreasing LR to {new_lr:.2e} (confidence trend: {slope:.4f})")
                    
            else:  # Confidence is relatively stable
                # Keep current LR
                adjustment_result["recommendation"] = "maintain_lr"
                
        return adjustment_result
        
    def get_status(self) -> Dict[str, Any]:
        """Get controller status"""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "window_size": self.window_size,
            "adjustment_factor": self.adjustment_factor,
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
            "confidence_history_length": len(self.confidence_history),
            "lr_history_length": len(self.lr_history)
        }


class UnifiedControllerInterface:
    """Main unified interface that consolidates all controllers"""
    
    def __init__(self):
        # Initialize all controllers
        self.extend_stop = ExtendStopController()
        self.predictive_stop = PredictiveStopController()
        self.meta_lr_policy = MetaLRPolicyController()
        self.confidence_trend_lr = ConfidenceTrendBasedLRAdjuster()
        
        # Link controllers
        self.extend_stop.set_predictive_forecaster(self.predictive_stop)
        
        # Controller registry
        self.controllers = {
            "extend_stop": self.extend_stop,
            "predictive_stop": self.predictive_stop,
            "meta_lr_policy": self.meta_lr_policy,
            "confidence_trend_lr": self.confidence_trend_lr
        }
        
        # Link status
        self.links = {
            "extend_stop->predictive_stop": True
        }
        
    def get_controller(self, name: str) -> Optional[BaseController]:
        """Get controller by name"""
        return self.controllers.get(name)
        
    def enable_controller(self, name: str):
        """Enable controller by name"""
        if name in self.controllers:
            self.controllers[name].enable()
            
    def disable_controller(self, name: str):
        """Disable controller by name"""
        if name in self.controllers:
            self.controllers[name].disable()
            
    def get_all_controllers(self) -> Dict[str, BaseController]:
        """Get all controllers"""
        return self.controllers.copy()
        
    def get_controller_status(self) -> Dict[str, Any]:
        """Get status of all controllers"""
        status = {}
        for name, controller in self.controllers.items():
            if hasattr(controller, 'get_status'):
                status[name] = controller.get_status()
            else:
                status[name] = {"name": controller.get_name(), "enabled": controller.is_enabled()}
        return status
        
    def get_links_status(self) -> Dict[str, bool]:
        """Get controller linking status"""
        return self.links.copy()
        
    def reset_all_controllers(self):
        """Reset all controllers"""
        for controller in self.controllers.values():
            controller.reset()
            
    def update_all_controllers(self, **kwargs):
        """Update all controllers with current training state"""
        # This method would be called during training to update all controllers
        # with the current state of training
        pass
        
    def get_unified_recommendation(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get unified recommendation from all active controllers
        
        Args:
            current_state: Dictionary containing current training state
                - loss: current loss value
                - metric: current validation metric
                - epoch: current epoch
                - step: current step
                - lr: current learning rate
                - confidence: current model confidence (if available)
                
        Returns:
            dict: Unified recommendation from all controllers
        """
        recommendations = {
            "extend_stop": None,
            "predictive_stop": None,
            "meta_lr_policy": None,
            "confidence_trend_lr": None,
            "unified_action": {
                "stop_training": False,
                "adjust_lr": False,
                "save_checkpoint": False,
                "lr_change_factor": 1.0,
                "reasons": []
            }
        }
        
        # Get ExtendStop recommendation
        if self.extend_stop.is_enabled():
            extend_stop_result = self.extend_stop(
                current_state.get("loss", 0.0),
                current_state.get("metric", 0.0),
                current_state.get("epoch", 0)
            )
            recommendations["extend_stop"] = extend_stop_result
            
            # Update unified action based on ExtendStop
            if extend_stop_result["action"] == "stop":
                recommendations["unified_action"]["stop_training"] = True
                recommendations["unified_action"]["reasons"].append("ExtendStop recommends stopping")
                
            if extend_stop_result["reduce_lr"]:
                recommendations["unified_action"]["adjust_lr"] = True
                recommendations["unified_action"]["lr_change_factor"] *= 0.5
                recommendations["unified_action"]["reasons"].append("ExtendStop recommends reducing LR")
                
            if extend_stop_result["save_checkpoint"]:
                recommendations["unified_action"]["save_checkpoint"] = True
                recommendations["unified_action"]["reasons"].append("ExtendStop recommends saving checkpoint")
        
        # Get PredictiveStop recommendation
        if self.predictive_stop.is_enabled():
            predictive_result = self.predictive_stop.predict_saturation(
                current_state.get("loss", 0.0),
                current_state.get("metric", 0.0),
                current_state.get("epoch", 0)
            )
            recommendations["predictive_stop"] = predictive_result
            
            # Update unified action based on PredictiveStop
            if (predictive_result["saturation_predicted"] and 
                predictive_result["confidence"] > 0.7 and
                predictive_result["epochs_until_saturation"] is not None and
                predictive_result["epochs_until_saturation"] <= 2):
                recommendations["unified_action"]["stop_training"] = True
                recommendations["unified_action"]["reasons"].append("PredictiveStop forecasts saturation")
        
        # Get MetaLRPolicy recommendation
        if self.meta_lr_policy.is_enabled():
            loss_improvement = current_state.get("loss_improvement", 0.0)
            meta_lr_result = self.meta_lr_policy.get_lr_action(
                current_state.get("loss", 0.0),
                loss_improvement,
                current_state.get("lr", 1e-3),
                current_state.get("step", 0)
            )
            recommendations["meta_lr_policy"] = meta_lr_result
            
            # Update unified action based on MetaLRPolicy
            if meta_lr_result["action"] == "reduce_lr":
                recommendations["unified_action"]["adjust_lr"] = True
                recommendations["unified_action"]["lr_change_factor"] *= 0.8
                recommendations["unified_action"]["reasons"].append("MetaLRPolicy recommends reducing LR")
            elif meta_lr_result["action"] == "increase_lr":
                recommendations["unified_action"]["adjust_lr"] = True
                recommendations["unified_action"]["lr_change_factor"] *= 1.2
                recommendations["unified_action"]["reasons"].append("MetaLRPolicy recommends increasing LR")
        
        # Get ConfidenceTrendLR recommendation
        if self.confidence_trend_lr.is_enabled() and "confidence" in current_state:
            # Note: This would typically be called separately as it modifies the optimizer directly
            pass
            
        return recommendations


# Example usage
if __name__ == "__main__":
    # Create unified controller interface
    controller_interface = UnifiedControllerInterface()
    
    # Example training loop integration
    print("Unified Controller Interface for MAHIA-X")
    print("=" * 40)
    
    # Simulate training state
    training_state = {
        "loss": 0.5,
        "metric": 0.8,
        "epoch": 10,
        "step": 100,
        "lr": 1e-3,
        "confidence": 0.75,
        "loss_improvement": -0.01
    }
    
    # Get unified recommendation
    recommendation = controller_interface.get_unified_recommendation(training_state)
    
    print("Controller Status:")
    status = controller_interface.get_controller_status()
    for controller_name, controller_status in status.items():
        print(f"  {controller_name}: {'Enabled' if controller_status.get('enabled', True) else 'Disabled'}")
    
    print("\nUnified Recommendation:")
    unified_action = recommendation["unified_action"]
    print(f"  Stop Training: {unified_action['stop_training']}")
    print(f"  Adjust LR: {unified_action['adjust_lr']}")
    print(f"  LR Change Factor: {unified_action['lr_change_factor']:.2f}")
    print(f"  Save Checkpoint: {unified_action['save_checkpoint']}")
    print(f"  Reasons: {', '.join(unified_action['reasons']) if unified_action['reasons'] else 'None'}")
    
    print("\nIndividual Controller Recommendations:")
    for controller_name, rec in recommendation.items():
        if controller_name != "unified_action" and rec is not None:
            print(f"  {controller_name}: {rec}")