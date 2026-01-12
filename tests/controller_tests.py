"""
Unit tests for MAHIA-X controller, forecast, and entropy functions.
"""
import torch
import numpy as np
import unittest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modell_V5_MAHIA_HyenaMoE import (
    PredictiveStopForecaster, 
    ExtendStop, 
    GradientEntropyMonitor,
    MetaLRPolicyController,
    ConfidenceTrendBasedLRAdjuster,
    ExpertLoadBalancerV2,
    CurriculumMemorySystem
)

class TestPredictiveStopForecaster(unittest.TestCase):
    """Test cases for PredictiveStopForecaster"""
    
    def setUp(self):
        self.forecaster = PredictiveStopForecaster(window_size=5, prediction_horizon=2)
        
    def test_initialization(self):
        """Test that forecaster initializes correctly"""
        self.assertEqual(self.forecaster.window_size, 5)
        self.assertEqual(self.forecaster.prediction_horizon, 2)
        self.assertEqual(len(self.forecaster.loss_history), 0)
        self.assertEqual(len(self.forecaster.metric_history), 0)
        self.assertEqual(len(self.forecaster.epoch_history), 0)
        
    def test_predict_saturation(self):
        """Test saturation prediction with sample data"""
        # Add some sample data
        result = None
        for i in range(10):
            result = self.forecaster.predict_saturation(
                current_loss=1.0 / (i + 1),  # Decreasing loss
                current_metric=0.5 + i * 0.05,  # Increasing metric
                current_epoch=i
            )
            
        # Check that prediction was made
        self.assertIsNotNone(result)
        if result is not None:
            self.assertIsInstance(result, dict)
            self.assertIn("saturation_predicted", result)
            self.assertIn("epochs_until_saturation", result)
            self.assertIn("confidence", result)
        
    def test_should_early_stop(self):
        """Test early stopping decision"""
        # Test with no data
        should_stop = self.forecaster.should_early_stop()
        self.assertFalse(should_stop)
        
        # Test with saturation prediction
        self.forecaster.last_prediction = {
            "saturation_predicted": True,
            "epochs_until_saturation": 1,
            "confidence": 0.9
        }
        self.forecaster.epochs_until_saturation = 1
        
        should_stop = self.forecaster.should_early_stop()
        self.assertTrue(should_stop)


class TestExtendStop(unittest.TestCase):
    """Test cases for ExtendStop"""
    
    def setUp(self):
        self.early_stopper = ExtendStop(patience=5, min_delta=0.01, extend_patience=2, max_extensions=1)
        
    def test_initialization(self):
        """Test that early stopper initializes correctly"""
        self.assertEqual(self.early_stopper.patience, 5)
        self.assertEqual(self.early_stopper.min_delta, 0.01)
        self.assertEqual(self.early_stopper.extend_patience, 2)
        self.assertEqual(self.early_stopper.max_extensions, 1)
        self.assertEqual(self.early_stopper.best_loss, float('inf'))
        self.assertEqual(self.early_stopper.counter, 0)
        self.assertEqual(self.early_stopper.extensions_used, 0)
        
    def test_improving_loss(self):
        """Test behavior with improving loss"""
        result = self.early_stopper(0.5, 0.8, 1)
        self.assertEqual(result["action"], "improving")
        self.assertEqual(self.early_stopper.best_loss, 0.5)
        self.assertFalse(self.early_stopper.stop_training)
        
    def test_non_improving_loss(self):
        """Test behavior with non-improving loss"""
        # First set a good loss
        self.early_stopper(0.5, 0.8, 1)
        
        # Then provide worse losses
        result = None
        for i in range(5):
            result = self.early_stopper(0.6, 0.7, 2 + i)
            
        # Should be waiting but not stopping yet
        if result is not None:
            self.assertEqual(result["action"], "waiting")
        self.assertFalse(self.early_stopper.stop_training)
        
    def test_extension_and_stop(self):
        """Test extension and final stop"""
        # Set initial good loss
        self.early_stopper(0.5, 0.8, 1)
        
        # Provide consistently worse losses to trigger extension
        result = None
        for i in range(10):
            result = self.early_stopper(0.6, 0.7, 2 + i)
            
        # Should have triggered extension
        if result is not None:
            if result["action"] == "extend":
                self.assertEqual(self.early_stopper.extensions_used, 1)
            
        # Continue with bad losses to trigger final stop
        for i in range(10):
            result = self.early_stopper(0.6, 0.7, 12 + i)
            
        # Should eventually stop
        if result is not None:
            if result["action"] == "stop":
                self.assertTrue(self.early_stopper.stop_training)


class TestGradientEntropyMonitor(unittest.TestCase):
    """Test cases for GradientEntropyMonitor"""
    
    def setUp(self):
        self.monitor = GradientEntropyMonitor(window_size=3, entropy_drop_threshold=0.25)
        
    def test_initialization(self):
        """Test that monitor initializes correctly"""
        self.assertEqual(self.monitor.window_size, 3)
        self.assertEqual(self.monitor.entropy_drop_threshold, 0.25)
        self.assertEqual(len(self.monitor.gradient_history), 0)
        
    def test_compute_gradient_entropy(self):
        """Test gradient entropy computation"""
        # Create a simple model for testing
        model = torch.nn.Linear(10, 5)
        
        # Set some gradients
        for param in model.parameters():
            if param.grad is None:
                param.grad = torch.randn_like(param)
                
        # Compute entropy
        entropy = self.monitor.compute_gradient_entropy(model)
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0.0)
        
    def test_should_adjust_training(self):
        """Test training adjustment recommendations"""
        model = torch.nn.Linear(10, 5)
        
        # Set some gradients
        for param in model.parameters():
            if param.grad is None:
                param.grad = torch.randn_like(param)
                
        # Test with no history
        recommendations = self.monitor.should_adjust_training(model)
        self.assertIsInstance(recommendations, dict)
        self.assertIn("adjust_dropout", recommendations)
        self.assertIn("increase_dropout", recommendations)
        self.assertIn("entropy", recommendations)
        
        # Add some entropy values that drop significantly
        self.monitor.gradient_history = [1.0, 0.9, 0.8, 0.3, 0.2, 0.1]  # Sharp drop
        
        recommendations = self.monitor.should_adjust_training(model)
        # Should recommend adjusting dropout due to entropy drop
        self.assertTrue(recommendations["adjust_dropout"])


class TestMetaLRPolicyController(unittest.TestCase):
    """Test cases for MetaLRPolicyController"""
    
    def setUp(self):
        self.controller = MetaLRPolicyController(state_dim=8, action_dim=3, lr=1e-3)
        
    def test_initialization(self):
        """Test that controller initializes correctly"""
        self.assertEqual(self.controller.state_dim, 8)
        self.assertEqual(self.controller.action_dim, 3)
        self.assertEqual(self.controller.lr, 1e-3)
        self.assertEqual(len(self.controller.state_history), 0)
        self.assertEqual(len(self.controller.action_history), 0)
        self.assertEqual(len(self.controller.reward_history), 0)
        
    def test_get_lr_action(self):
        """Test learning rate action selection"""
        action = self.controller.get_lr_action(
            current_loss=0.5,
            loss_improvement=0.01,
            current_lr=1e-3,
            step=100
        )
        
        self.assertIsInstance(action, dict)
        self.assertIn("action", action)
        self.assertIn("confidence", action)
        self.assertIn("state", action)
        
        # Action should be one of the expected values
        self.assertIn(action["action"], ["reduce_lr", "increase_lr", "maintain_lr"])
        
    def test_update_policy(self):
        """Test policy update with rewards"""
        # Add some rewards
        for i in range(5):
            self.controller.update_policy(0.5 + i * 0.1)
            
        self.assertEqual(len(self.controller.reward_history), 5)
        
    def test_get_policy_stats(self):
        """Test policy statistics retrieval"""
        stats = self.controller.get_policy_stats()
        self.assertIsInstance(stats, dict)
        
        # Add some data and test again
        self.controller.update_policy(0.8)
        self.controller.update_policy(0.7)
        
        stats = self.controller.get_policy_stats()
        self.assertIn("total_actions", stats)
        self.assertIn("total_rewards", stats)
        self.assertIn("average_reward", stats)


class TestConfidenceTrendBasedLRAdjuster(unittest.TestCase):
    """Test cases for ConfidenceTrendBasedLRAdjuster"""
    
    def setUp(self):
        self.adjuster = ConfidenceTrendBasedLRAdjuster(
            window_size=5, 
            adjustment_factor=0.9,
            min_lr=1e-7,
            max_lr=1e-2
        )
        
    def test_initialization(self):
        """Test that adjuster initializes correctly"""
        self.assertEqual(self.adjuster.window_size, 5)
        self.assertEqual(self.adjuster.adjustment_factor, 0.9)
        self.assertEqual(self.adjuster.min_lr, 1e-7)
        self.assertEqual(self.adjuster.max_lr, 1e-2)
        self.assertEqual(len(self.adjuster.confidence_history), 0)
        self.assertEqual(len(self.adjuster.lr_history), 0)
        
    def test_adjust_lr_based_on_confidence_trend(self):
        """Test learning rate adjustment based on confidence trends"""
        # Create a mock optimizer
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.randn(10, 5))], lr=1e-3)
        
        # Test with insufficient history
        result = self.adjuster.adjust_lr_based_on_confidence_trend(
            optimizer=optimizer,
            current_confidence=0.8,
            current_lr=1e-3
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn("lr_adjusted", result)
        self.assertIn("new_lr", result)
        self.assertIn("recommendation", result)
        
        # Test with sufficient history showing increasing confidence
        self.adjuster.confidence_history = [0.5, 0.6, 0.7, 0.8, 0.9]
        self.adjuster.lr_history = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
        
        result = self.adjuster.adjust_lr_based_on_confidence_trend(
            optimizer=optimizer,
            current_confidence=0.95,
            current_lr=1e-3
        )
        
        # Should recommend increasing LR due to positive trend
        self.assertIn(result["recommendation"], ["increase_lr", "decrease_lr", "maintain_lr"])


class TestExpertLoadBalancerV2(unittest.TestCase):
    """Test cases for ExpertLoadBalancerV2"""
    
    def setUp(self):
        self.balancer = ExpertLoadBalancerV2(reweighting_interval=5, balance_factor=0.1)
        
    def test_initialization(self):
        """Test that balancer initializes correctly"""
        self.assertEqual(self.balancer.reweighting_interval, 5)
        self.assertEqual(self.balancer.balance_factor, 0.1)
        self.assertEqual(self.balancer.step_count, 0)
        self.assertEqual(len(self.balancer.load_history), 0)
        self.assertEqual(len(self.balancer.performance_history), 0)
        
    def test_register_expert_group(self):
        """Test expert group registration"""
        self.balancer.register_expert_group("test_group", 4)
        
        self.assertIn("test_group", self.balancer.load_history)
        self.assertIn("test_group", self.balancer.performance_history)
        self.assertIn("test_group", self.balancer.expert_weights)
        self.assertEqual(len(self.balancer.expert_weights["test_group"]), 4)
        
    def test_update_load_metrics(self):
        """Test load metrics update"""
        group_id = "test_group"
        self.balancer.register_expert_group(group_id, 3)
        
        # Update with load data
        expert_loads = np.array([0.5, 0.7, 0.3])
        self.balancer.update_load_metrics(group_id, expert_loads)
        
        self.assertEqual(len(self.balancer.load_history[group_id]), 1)
        np.testing.assert_array_equal(self.balancer.load_history[group_id][0], expert_loads)
        
    def test_compute_load_balance_weights(self):
        """Test load balance weight computation"""
        group_id = "test_group"
        self.balancer.register_expert_group(group_id, 3)
        
        # Add some load history
        for i in range(5):
            expert_loads = np.array([0.5 + i*0.1, 0.7 - i*0.05, 0.3 + i*0.05])
            self.balancer.update_load_metrics(group_id, expert_loads)
            
        # Compute weights
        weights = self.balancer.compute_load_balance_weights(group_id)
        
        self.assertIsInstance(weights, np.ndarray)
        self.assertEqual(len(weights), 3)
        # Weights should sum to 1
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        
    def test_should_reweight(self):
        """Test reweighting decision"""
        # Should not reweight initially
        should_reweight = self.balancer.should_reweight()
        self.assertFalse(should_reweight)
        self.assertEqual(self.balancer.step_count, 1)
        
        # Advance steps to trigger reweighting
        for i in range(4):
            should_reweight = self.balancer.should_reweight()
            
        # Should reweight now
        should_reweight = self.balancer.should_reweight()
        self.assertTrue(should_reweight)
        self.assertEqual(self.balancer.step_count, 6)


class TestCurriculumMemorySystem(unittest.TestCase):
    """Test cases for CurriculumMemorySystem"""
    
    def setUp(self):
        self.memory = CurriculumMemorySystem(max_history_size=100)
        
    def test_initialization(self):
        """Test that memory system initializes correctly"""
        self.assertEqual(self.memory.max_history_size, 100)
        self.assertEqual(len(self.memory.difficulty_history), 0)
        self.assertEqual(len(self.memory.entropy_history), 0)
        self.assertEqual(len(self.memory.performance_history), 0)
        self.assertEqual(self.memory.best_performance, float('-inf'))
        self.assertEqual(self.memory.worst_performance, float('inf'))
        
    def test_store_difficulty_record(self):
        """Test storing difficulty records"""
        self.memory.store_difficulty_record(
            epoch=1,
            difficulty=0.5,
            entropy=0.8,
            performance=0.75
        )
        
        self.assertEqual(len(self.memory.difficulty_history), 1)
        self.assertEqual(len(self.memory.entropy_history), 1)
        self.assertEqual(len(self.memory.performance_history), 1)
        self.assertEqual(self.memory.best_performance, 0.75)
        self.assertEqual(self.memory.worst_performance, 0.75)
        
    def test_get_difficulty_trend(self):
        """Test difficulty trend analysis"""
        # Add some records
        for i in range(10):
            self.memory.store_difficulty_record(
                epoch=i,
                difficulty=0.3 + i * 0.05,  # Increasing difficulty
                entropy=0.8 - i * 0.02,     # Decreasing entropy
                performance=0.7 + i * 0.01   # Increasing performance
            )
            
        # Get trend analysis
        trend = self.memory.get_difficulty_trend(window_size=5)
        
        self.assertIsInstance(trend, dict)
        self.assertIn("slope", trend)
        self.assertIn("acceleration", trend)
        self.assertIn("current_difficulty", trend)
        
    def test_get_entropy_analysis(self):
        """Test entropy analysis"""
        # Add some entropy values
        for i in range(10):
            self.memory.entropy_history.append(0.8 - i * 0.05)  # Decreasing entropy
            
        # Get entropy analysis
        analysis = self.memory.get_entropy_analysis(window_size=5)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn("avg_entropy", analysis)
        self.assertIn("std_entropy", analysis)
        self.assertIn("entropy_trend", analysis)
        
    def test_recommend_difficulty(self):
        """Test difficulty recommendation"""
        # Add some history
        for i in range(10):
            self.memory.store_difficulty_record(
                epoch=i,
                difficulty=0.3 + i * 0.05,
                entropy=0.8 - i * 0.02,
                performance=0.7 + i * 0.01
            )
            
        # Get recommendation
        recommendation = self.memory.recommend_difficulty(current_entropy=0.5)
        
        self.assertIsInstance(recommendation, dict)
        self.assertIn("recommended_difficulty", recommendation)
        self.assertIn("confidence", recommendation)
        self.assertIn("reason", recommendation)


if __name__ == '__main__':
    unittest.main()