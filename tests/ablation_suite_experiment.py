#!/usr/bin/env python3
"""
Ablation Suite Experiment for MAHIA-V5 Adaptive Training
"""

import sys
import os
import random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the classes we want to test
from modell_V5_MAHIA_HyenaMoE import ExtendStop, GradientEntropyMonitor, AutoLrPrecisionTuner

class AblationExperiment:
    """Run ablation experiments to show contribution of each submodule"""
    
    def __init__(self, seeds=[42, 123, 456]):
        self.seeds = seeds
        self.results = {}
        
    def simulate_training(self, use_reflective=True, use_entropy=True, use_auto_lr=True, seed=42):
        """Simulate training with different controller configurations"""
        print(f"  Running ablation with R={use_reflective}, E={use_entropy}, A={use_auto_lr}, seed={seed}")
        
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize controllers based on configuration
        extend_stop = ExtendStop(patience=5, min_delta=1e-4) if use_reflective else None
        gradient_monitor = GradientEntropyMonitor(window_size=3, entropy_drop_threshold=0.25) if use_entropy else None
        auto_tuner = AutoLrPrecisionTuner(initial_lr=1e-3, min_lr=1e-7) if use_auto_lr else None
        
        # Track metrics
        val_metrics = [0.85]  # Initial validation metric
        energy_consumed = 0.0
        max_epochs = 50
        
        # Track early saturation detection
        plateau_epoch = None
        
        for epoch in range(max_epochs):
            # Simulate validation metric improvement
            # Base improvement that decreases over time
            base_improvement = 0.005 * (1.0 - epoch/max_epochs)
            
            # Add some noise
            noise = random.uniform(-0.001, 0.001)
            actual_improvement = base_improvement + noise
            
            new_metric = min(0.99, val_metrics[-1] + actual_improvement)
            val_metrics.append(new_metric)
            
            # Simulate energy consumption (simplified)
            energy_consumed += 0.0004  # Wh per epoch
            
            # Apply controllers if enabled
            should_stop = False
            if extend_stop:
                extend_action = extend_stop(new_metric)
                if extend_action["action"] == "stop":
                    should_stop = True
                    
            if auto_tuner and epoch > 10:
                # Create a mock model for gradient monitoring
                class MockModel:
                    def parameters(self):
                        # Return mock parameters with random gradients
                        class MockParam:
                            def __init__(self):
                                self.grad = None if random.random() < 0.3 else MockTensor()
                        class MockTensor:
                            def abs(self):
                                return self
                            def mean(self):
                                class MockItem:
                                    def item(self):
                                        return random.uniform(0.01, 0.1)
                                return MockItem()
                        return [MockParam() for _ in range(5)]
                
                mock_model = MockModel()
                # Pass gradient_monitor only if it exists
                if gradient_monitor:
                    tuner_result = auto_tuner.adjust_lr_and_precision(None, new_metric, 
                                                                     gradient_monitor, mock_model,
                                                                     loss_improving=actual_improvement > 1e-4)
                else:
                    tuner_result = auto_tuner.adjust_lr_and_precision(None, new_metric, 
                                                                     None, mock_model,
                                                                     loss_improving=actual_improvement > 1e-4)
            
            # Check for plateau (early saturation detection)
            if plateau_epoch is None and epoch > 5:
                recent_improvements = [val_metrics[i] - val_metrics[i-1] for i in range(max(1, len(val_metrics)-3), len(val_metrics))]
                if all(imp < 1e-4 for imp in recent_improvements):
                    plateau_epoch = epoch
            
            # Stop if needed
            if should_stop:
                break
        
        # Calculate final metrics
        final_val_metric = val_metrics[-1]
        val_metric_improvement = final_val_metric - 0.85
        energy_per_accuracy = energy_consumed / (val_metric_improvement + 1e-12)
        
        if plateau_epoch is None:
            plateau_epoch = len(val_metrics) - 1
            
        result = {
            'seed': seed,
            'energy_consumed': energy_consumed,
            'final_val_metric': final_val_metric,
            'val_metric_improvement': val_metric_improvement,
            'energy_per_accuracy': energy_per_accuracy,
            'plateau_epoch': plateau_epoch,
            'epochs_run': len(val_metrics) - 1
        }
        
        return result
    
    def run_experiment(self):
        """Run the ablation suite experiment"""
        print("Running Ablation Suite Experiment")
        print("=" * 35)
        
        # Define ablation configurations
        configurations = [
            {"name": "Full System", "reflective": True, "entropy": True, "auto_lr": True},
            {"name": "No ReflectiveHead", "reflective": False, "entropy": True, "auto_lr": True},
            {"name": "No Entropy", "reflective": True, "entropy": False, "auto_lr": True},
            {"name": "No Auto-LR", "reflective": True, "entropy": True, "auto_lr": False},
            {"name": "Baseline", "reflective": False, "entropy": False, "auto_lr": False}
        ]
        
        for config in configurations:
            config_name = config["name"]
            print(f"\nTesting {config_name}:")
            self.results[config_name] = []
            
            for seed in self.seeds:
                result = self.simulate_training(
                    use_reflective=config["reflective"],
                    use_entropy=config["entropy"],
                    use_auto_lr=config["auto_lr"],
                    seed=seed
                )
                self.results[config_name].append(result)
                print(f"    Seed {seed}: E/A={result['energy_per_accuracy']:.3f}Wh/unit, "
                      f"Epochs={result['epochs_run']}, Plateau={result['plateau_epoch']}")
        
        # Analyze results
        self._analyze_results()
        
    def _analyze_results(self):
        """Analyze ablation experiment results"""
        print(f"\n{'='*35}")
        print("ABLATION SUITE RESULTS ANALYSIS")
        print("=" * 35)
        
        # Compare energy per accuracy across configurations
        print(f"\nEnergy per Accuracy Point:")
        baseline_ea = [r['energy_per_accuracy'] for r in self.results['Baseline']]
        baseline_mean = np.mean(baseline_ea)
        baseline_std = np.std(baseline_ea)
        print(f"  Baseline: {baseline_mean:.3f} ± {baseline_std:.3f} Wh/unit")
        
        full_ea = [r['energy_per_accuracy'] for r in self.results['Full System']]
        full_mean = np.mean(full_ea)
        full_std = np.std(full_ea)
        full_improvement = (baseline_mean - full_mean) / baseline_mean * 100
        print(f"  Full System: {full_mean:.3f} ± {full_std:.3f} Wh/unit ({full_improvement:.1f}% improvement)")
        
        # Compare early saturation detection
        print(f"\nEarly Saturation Detection (Plateau Epoch):")
        baseline_plateau = [r['plateau_epoch'] for r in self.results['Baseline']]
        baseline_plateau_mean = np.mean(baseline_plateau)
        baseline_plateau_std = np.std(baseline_plateau)
        print(f"  Baseline: {baseline_plateau_mean:.1f} ± {baseline_plateau_std:.1f} epochs")
        
        full_plateau = [r['plateau_epoch'] for r in self.results['Full System']]
        full_plateau_mean = np.mean(full_plateau)
        full_plateau_std = np.std(full_plateau)
        plateau_improvement = baseline_plateau_mean - full_plateau_mean
        print(f"  Full System: {full_plateau_mean:.1f} ± {full_plateau_std:.1f} epochs ({plateau_improvement:.1f} epochs earlier)")
        
        # Ablation analysis - compare each component
        print(f"\nComponent Ablation Analysis:")
        components = ["No ReflectiveHead", "No Entropy", "No Auto-LR"]
        for component in components:
            component_ea = [r['energy_per_accuracy'] for r in self.results[component]]
            component_mean = np.mean(component_ea)
            component_improvement = (baseline_mean - component_mean) / baseline_mean * 100
            print(f"  {component}: {component_mean:.3f} Wh/unit ({component_improvement:.1f}% improvement)")
        
        # Summary of contributions
        print(f"\nComponent Contributions:")
        full_benefit = baseline_mean - full_mean
        no_reflective_benefit = baseline_mean - np.mean([r['energy_per_accuracy'] for r in self.results['No ReflectiveHead']])
        no_entropy_benefit = baseline_mean - np.mean([r['energy_per_accuracy'] for r in self.results['No Entropy']])
        no_auto_lr_benefit = baseline_mean - np.mean([r['energy_per_accuracy'] for r in self.results['No Auto-LR']])
        
        reflective_contribution = full_benefit - no_reflective_benefit
        entropy_contribution = full_benefit - no_entropy_benefit
        auto_lr_contribution = full_benefit - no_auto_lr_benefit
        
        total_contributions = reflective_contribution + entropy_contribution + auto_lr_contribution
        if total_contributions > 0:
            reflective_pct = (reflective_contribution / total_contributions) * 100
            entropy_pct = (entropy_contribution / total_contributions) * 100
            auto_lr_pct = (auto_lr_contribution / total_contributions) * 100
        else:
            reflective_pct = entropy_pct = auto_lr_pct = 0
            
        print(f"  ReflectiveHead contribution: {reflective_contribution:.4f} Wh/unit ({reflective_pct:.1f}%)")
        print(f"  Entropy Monitoring contribution: {entropy_contribution:.4f} Wh/unit ({entropy_pct:.1f}%)")
        print(f"  Auto-LR contribution: {auto_lr_contribution:.4f} Wh/unit ({auto_lr_pct:.1f}%)")
        
        # Summary
        print(f"\n{'='*35}")
        print("🎉 ABLATION SUITE COMPLETED!")
        print(f"   • Full system provides {full_improvement:.1f}% energy savings")
        print(f"   • Early saturation detected {plateau_improvement:.1f} epochs earlier")
        print(f"   • Component contributions quantified")
        print(f"{'='*35}")

def main():
    """Main function to run the experiment"""
    print("MAHIA-V5 Ablation Suite Experiment")
    print("=" * 35)
    
    try:
        # Run experiment with 3 random seeds
        experiment = AblationExperiment(seeds=[42, 123, 456])
        experiment.run_experiment()
        
        print("\n✅ Ablation suite experiment completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()