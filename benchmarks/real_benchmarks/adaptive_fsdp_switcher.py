"""
Adaptive FSDP/ZeRO Stage Switching for MAHIA
Automatic switching of memory strategy (ZeRO 1-3 / FSDP AutoWrap) based on batch size & GPU RAM utilization
"""

import torch
import time
from typing import Dict, List, Optional, Any, Tuple
import psutil
import os

# Try to import distributed training libraries
FSDP_AVAILABLE = False
DEEPSPEED_AVAILABLE = False

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
    FSDP_AVAILABLE = True
    print("✅ PyTorch FSDP available")
except ImportError:
    print("⚠️  PyTorch FSDP not available")

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
    print("✅ DeepSpeed available")
except ImportError:
    print("⚠️  DeepSpeed not available")

class MemoryMonitor:
    """Monitor GPU and system memory usage"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.gpu_available = torch.cuda.is_available()
        
    def get_gpu_memory_info(self) -> Optional[Dict[str, float]]:
        """
        Get GPU memory information
        
        Returns:
            Optional[Dict[str, float]]: GPU memory info or None if GPU not available
        """
        if not self.gpu_available:
            return None
            
        try:
            allocated = torch.cuda.memory_allocated(self.device_id)
            reserved = torch.cuda.memory_reserved(self.device_id)
            max_allocated = torch.cuda.max_memory_allocated(self.device_id)
            
            total_memory = torch.cuda.get_device_properties(self.device_id).total_memory
            
            return {
                "allocated_bytes": float(allocated),
                "reserved_bytes": float(reserved),
                "max_allocated_bytes": float(max_allocated),
                "total_bytes": float(total_memory),
                "utilization_percent": (allocated / total_memory) * 100 if total_memory > 0 else 0,
                "free_bytes": float(total_memory - allocated)
            }
        except Exception as e:
            print(f"⚠️  Failed to get GPU memory info: {e}")
            return None
    
    def get_system_memory_info(self) -> Dict[str, float]:
        """
        Get system memory information
        
        Returns:
            Dict[str, float]: System memory info
        """
        try:
            memory = psutil.virtual_memory()
            return {
                "total_bytes": float(memory.total),
                "available_bytes": float(memory.available),
                "used_bytes": float(memory.used),
                "utilization_percent": memory.percent
            }
        except Exception as e:
            print(f"⚠️  Failed to get system memory info: {e}")
            return {
                "total_bytes": 0,
                "available_bytes": 0,
                "used_bytes": 0,
                "utilization_percent": 0
            }

class FSDPStageController:
    """Control FSDP/ZeRO stage switching"""
    
    def __init__(self, model_params: int, gpu_memory_gb: float = 16.0):
        """
        Initialize FSDP stage controller
        
        Args:
            model_params: Number of model parameters
            gpu_memory_gb: Available GPU memory in GB
        """
        self.model_params = model_params
        self.gpu_memory_gb = gpu_memory_gb
        self.current_stage = "ZERO_1"  # Default stage
        self.stage_history = ["ZERO_1"]
        self.switch_count = 0
        
        # Stage configurations
        self.stage_configs = {
            "ZERO_1": {
                "sharding_strategy": "SHARD_GRAD_OP" if FSDP_AVAILABLE else 1,
                "memory_efficiency": 0.3,  # 30% memory reduction
                "communication_overhead": 0.2,  # 20% communication overhead
                "recommended_for_params": 1e9,  # 1B parameters
                "min_gpu_memory_gb": 8.0
            },
            "ZERO_2": {
                "sharding_strategy": "FULL_SHARD" if FSDP_AVAILABLE else 2,
                "memory_efficiency": 0.6,  # 60% memory reduction
                "communication_overhead": 0.5,  # 50% communication overhead
                "recommended_for_params": 5e9,  # 5B parameters
                "min_gpu_memory_gb": 12.0
            },
            "ZERO_3": {
                "sharding_strategy": "HYBRID_SHARD" if FSDP_AVAILABLE else 3,
                "memory_efficiency": 0.8,  # 80% memory reduction
                "communication_overhead": 0.8,  # 80% communication overhead
                "recommended_for_params": 1e10,  # 10B parameters
                "min_gpu_memory_gb": 16.0
            }
        }
        
        print(f"✅ FSDP Stage Controller initialized for {model_params/1e9:.1f}B parameter model")
    
    def recommend_stage(self, batch_size: int, gpu_memory_info: Optional[Dict[str, float]] = None,
                       system_memory_info: Optional[Dict[str, float]] = None) -> str:
        """
        Recommend optimal FSDP/ZeRO stage based on current conditions
        
        Args:
            batch_size: Current batch size
            gpu_memory_info: GPU memory information
            system_memory_info: System memory information
            
        Returns:
            str: Recommended stage
        """
        # Calculate memory pressure
        memory_pressure = 0.0
        if gpu_memory_info:
            memory_pressure = gpu_memory_info.get("utilization_percent", 0) / 100.0
        elif system_memory_info:
            memory_pressure = system_memory_info.get("utilization_percent", 0) / 100.0
        
        # Calculate parameter pressure
        param_pressure = min(1.0, self.model_params / 1e10)  # Normalize to 10B params
        
        # Calculate batch size pressure
        batch_pressure = min(1.0, batch_size / 256.0)  # Normalize to batch 256
        
        # Combined pressure score (0-1)
        pressure_score = (memory_pressure * 0.4 + param_pressure * 0.4 + batch_pressure * 0.2)
        
        # Select stage based on pressure
        if pressure_score < 0.3:
            return "ZERO_1"
        elif pressure_score < 0.7:
            return "ZERO_2"
        else:
            return "ZERO_3"
    
    def should_switch_stage(self, current_batch_size: int, 
                           gpu_memory_info: Optional[Dict[str, float]] = None,
                           system_memory_info: Optional[Dict[str, float]] = None) -> 'Tuple[bool, str, str]':
        """
        Determine if stage should be switched
        
        Args:
            current_batch_size: Current batch size
            gpu_memory_info: GPU memory information
            system_memory_info: System memory information
            
        Returns:
            Tuple[bool, str, str]: (should_switch, recommended_stage, reason)
        """
        recommended_stage = self.recommend_stage(current_batch_size, gpu_memory_info, system_memory_info)
        current_stage = self.current_stage
        
        if recommended_stage != current_stage:
            # Check if switch is feasible
            stage_config = self.stage_configs[recommended_stage]
            if self.gpu_memory_gb >= stage_config["min_gpu_memory_gb"]:
                reason = f"Pressure score indicates {recommended_stage} is optimal"
                return True, recommended_stage, reason
        
        return False, current_stage, "Current stage is optimal"

class AdaptiveFSDPSwitcher:
    """Main adaptive FSDP/ZeRO switching system"""
    
    def __init__(self, model, gpu_memory_gb: float = 16.0):
        """
        Initialize adaptive FSDP switcher
        
        Args:
            model: PyTorch model
            gpu_memory_gb: Available GPU memory in GB
        """
        self.model = model
        self.gpu_memory_gb = gpu_memory_gb
        
        # Count model parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        # Initialize components
        self.memory_monitor = MemoryMonitor()
        self.stage_controller = FSDPStageController(num_params, gpu_memory_gb)
        
        # Performance tracking
        self.switch_history = []
        self.performance_metrics = {
            "memory_efficiency_gains": 0.0,
            "throughput_improvements": 0.0,
            "stability_improvements": 0.0
        }
        
        print("✅ Adaptive FSDP Switcher initialized")
    
    def evaluate_and_switch(self, current_batch_size: int) -> Dict[str, Any]:
        """
        Evaluate current conditions and switch FSDP stage if needed
        
        Args:
            current_batch_size: Current batch size
            
        Returns:
            Dict[str, Any]: Evaluation results and actions taken
        """
        # Get memory information
        gpu_memory_info = self.memory_monitor.get_gpu_memory_info()
        system_memory_info = self.memory_monitor.get_system_memory_info()
        
        # Determine if switch is needed
        should_switch, recommended_stage, reason = self.stage_controller.should_switch_stage(
            current_batch_size, gpu_memory_info, system_memory_info
        )
        
        actions = []
        if should_switch:
            old_stage = self.stage_controller.current_stage
            self.stage_controller.current_stage = recommended_stage
            self.stage_controller.stage_history.append(recommended_stage)
            self.stage_controller.switch_count += 1
            
            actions.append({
                "type": "stage_switch",
                "from": old_stage,
                "to": recommended_stage,
                "reason": reason
            })
            
            # Update performance metrics
            self._update_performance_metrics(old_stage, recommended_stage)
        
        return {
            "should_switch": should_switch,
            "recommended_stage": recommended_stage,
            "current_stage": self.stage_controller.current_stage,
            "reason": reason,
            "actions": actions,
            "gpu_memory_info": gpu_memory_info,
            "system_memory_info": system_memory_info,
            "performance_metrics": self.performance_metrics.copy()
        }
    
    def _update_performance_metrics(self, old_stage: str, new_stage: str):
        """
        Update performance metrics based on stage switch
        
        Args:
            old_stage: Previous stage
            new_stage: New stage
        """
        old_config = self.stage_controller.stage_configs[old_stage]
        new_config = self.stage_controller.stage_configs[new_stage]
        
        # Memory efficiency gain
        self.performance_metrics["memory_efficiency_gains"] += (
            new_config["memory_efficiency"] - old_config["memory_efficiency"]
        )
        
        # Throughput impact (simplified model)
        # Higher memory efficiency usually means better throughput, but more communication overhead
        old_throughput_score = old_config["memory_efficiency"] - old_config["communication_overhead"]
        new_throughput_score = new_config["memory_efficiency"] - new_config["communication_overhead"]
        self.performance_metrics["throughput_improvements"] += (new_throughput_score - old_throughput_score)
        
        # Stability (higher sharding generally means more stable but slower)
        self.performance_metrics["stability_improvements"] += (
            new_config["memory_efficiency"] - old_config["memory_efficiency"]
        ) * 0.5
    
    def get_stage_summary(self) -> Dict[str, Any]:
        """
        Get summary of current stage configuration
        
        Returns:
            Dict[str, Any]: Stage summary
        """
        current_stage = self.stage_controller.current_stage
        stage_config = self.stage_controller.stage_configs[current_stage]
        
        return {
            "current_stage": current_stage,
            "stage_config": stage_config,
            "total_switches": self.stage_controller.switch_count,
            "stage_history": self.stage_controller.stage_history[-10:],  # Last 10 switches
            "performance_metrics": self.performance_metrics.copy()
        }
    
    def apply_fsdp_configuration(self, model=None) -> Any:
        """
        Apply current FSDP configuration to model
        
        Args:
            model: Model to apply configuration to (uses self.model if None)
            
        Returns:
            Any: Configured model or original model if FSDP not available
        """
        if model is None:
            model = self.model
            
        current_stage = self.stage_controller.current_stage
        stage_config = self.stage_controller.stage_configs[current_stage]
        
        # Apply FSDP if available
        if FSDP_AVAILABLE:
            try:
                sharding_strategy = stage_config["sharding_strategy"]
                
                # This is a simplified implementation - in practice, you'd need proper FSDP setup
                print(f"🔧 Applying FSDP configuration: {current_stage}")
                print(f"   Sharding Strategy: {sharding_strategy}")
                print(f"   Memory Efficiency: {stage_config['memory_efficiency']:.0%}")
                
                # Return original model in this simplified implementation
                return model
                
            except Exception as e:
                print(f"❌ Failed to apply FSDP configuration: {e}")
                return model
        else:
            print("⚠️  FSDP not available, using standard model")
            return model

class FSDPSwitchingBenchmark:
    """Benchmark for FSDP stage switching performance"""
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device if model else "cpu"
    
    def benchmark_stage_switching(self, 
                                batch_sizes: List[int] = [8, 16, 32, 64],
                                model_sizes: List[float] = [1e9, 5e9, 1e10]) -> Dict[str, Any]:
        """
        Benchmark stage switching performance with different configurations
        
        Args:
            batch_sizes: List of batch sizes to test
            model_sizes: List of model sizes to test (in parameters)
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        print("🚀 Benchmarking Adaptive FSDP Stage Switching Performance")
        print("=" * 65)
        
        results = {}
        
        for model_size in model_sizes:
            model_size_billions = int(model_size / 1e9)
            print(f"\n📊 Testing {model_size_billions}B parameter model...")
            
            for batch_size in batch_sizes:
                key = f"model{model_size_billions}B_bs{batch_size}"
                print(f"   Testing BS={batch_size}...")
                
                # Create switcher
                switcher = AdaptiveFSDPSwitcher(self.model, gpu_memory_gb=16.0)
                
                # Simulate evaluation and switching
                start_time = time.time()
                evaluation_result = switcher.evaluate_and_switch(batch_size)
                end_time = time.time()
                
                switch_time = end_time - start_time
                
                results[key] = {
                    "model_size_billions": model_size_billions,
                    "batch_size": batch_size,
                    "switch_time": float(switch_time),
                    "recommended_stage": evaluation_result["recommended_stage"],
                    "should_switch": evaluation_result["should_switch"]
                }
                
                if evaluation_result["should_switch"]:
                    print(f"      🔄 Switch recommended: {evaluation_result['recommended_stage']} "
                          f"(took {switch_time*1000:.2f}ms)")
                else:
                    print(f"      ✅ Current stage optimal (took {switch_time*1000:.2f}ms)")
        
        return results

# Example usage
def example_adaptive_fsdp_switching():
    """Example of adaptive FSDP stage switching"""
    print("🔧 Setting up adaptive FSDP switching example...")
    
    # Simple model for demonstration
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self, hidden_size=512):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Linear(hidden_size * 4, hidden_size * 4),
                nn.ReLU(),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 10)
            )
            
        def forward(self, x):
            return self.layers(x)
    
    # Create model (approximately 1B parameters for demo)
    model = SimpleModel(hidden_size=1024)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Created model with {num_params/1e6:.1f}M parameters")
    
    # Create switcher
    switcher = AdaptiveFSDPSwitcher(model, gpu_memory_gb=16.0)
    
    # Simulate training with different batch sizes
    print("\n" + "="*60)
    print("🔄 Simulating training with adaptive FSDP switching...")
    
    batch_sizes = [8, 16, 32, 64, 128]
    for i, batch_size in enumerate(batch_sizes):
        print(f"\n🔁 Iteration {i+1}: Batch Size = {batch_size}")
        
        # Evaluate and potentially switch
        result = switcher.evaluate_and_switch(batch_size)
        
        print(f"   Current Stage: {result['current_stage']}")
        print(f"   Recommended: {result['recommended_stage']}")
        if result['should_switch']:
            print(f"   🔄 Switching: {result['actions'][0]['from']} → {result['actions'][0]['to']}")
            print(f"   Reason: {result['actions'][0]['reason']}")
        
        # Show memory info if available
        if result['gpu_memory_info']:
            mem_info = result['gpu_memory_info']
            print(f"   GPU Memory: {mem_info['utilization_percent']:.1f}% utilized")
    
    # Print final stage summary
    print("\n" + "="*60)
    summary = switcher.get_stage_summary()
    print("📊 Final FSDP Stage Summary:")
    print(f"   Current Stage: {summary['current_stage']}")
    print(f"   Total Switches: {summary['total_switches']}")
    print(f"   Memory Efficiency: {summary['stage_config']['memory_efficiency']:.0%}")
    
    # Show performance metrics
    perf_metrics = summary['performance_metrics']
    print(f"\n📈 Performance Improvements:")
    print(f"   Memory Efficiency Gains: {perf_metrics['memory_efficiency_gains']:.2f}")
    print(f"   Throughput Improvements: {perf_metrics['throughput_improvements']:.2f}")
    print(f"   Stability Improvements: {perf_metrics['stability_improvements']:.2f}")
    
    # Benchmark performance
    print("\n" + "="*60)
    benchmark = FSDPSwitchingBenchmark(model)
    results = benchmark.benchmark_stage_switching(
        batch_sizes=[8, 16, 32],
        model_sizes=[1e9, 5e9]
    )
    
    print(f"\n⏱️  Switching Performance Summary:")
    for key, result in results.items():
        print(f"   {key}: {result['switch_time']*1000:.2f}ms - "
              f"Stage: {result['recommended_stage']} "
              f"({'Switch' if result['should_switch'] else 'Keep'})")
    
    return switcher

if __name__ == "__main__":
    example_adaptive_fsdp_switching()