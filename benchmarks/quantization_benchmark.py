"""
Benchmark runner for comparing FP8/INT4 vs FP16 quantization performance.
"""
import torch
import time
import numpy as np
import sys
import os
from typing import Dict, Any, Tuple

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from modell_V5_MAHIA_HyenaMoE import (
        MAHIA_V5, 
        QATLoRAWrapper,
        FP8CalibrationAutoTuner
    )
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False
    print("⚠️  MAHIA-X modules not available for benchmarking")


class QuantizationBenchmarkRunner:
    """Benchmark runner for quantization comparisons"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def create_test_model(self, quantization_type: str = "fp16") -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Create a test model with specified quantization
        
        Args:
            quantization_type: Type of quantization ("fp16", "fp8", "int4", "fp32")
            
        Returns:
            Tuple of (model, metadata)
        """
        if not BENCHMARK_AVAILABLE:
            raise RuntimeError("MAHIA-X modules not available for benchmarking")
            
        # Create base model
        model = None
        if BENCHMARK_AVAILABLE:
            model = MAHIA_V5(
                vocab_size=1000,
                text_seq_len=32,
                tab_dim=20,
                embed_dim=32,
                fused_dim=64,
                moe_experts=4,
                moe_topk=2
            ).to(self.device)
        
        metadata = {
            "quantization_type": quantization_type,
            "model_params": sum(p.numel() for p in model.parameters()) if model is not None else 0,
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024) if model is not None else 0
        }
        
        # Apply quantization
        if model is not None:
            if quantization_type == "fp8" and BENCHMARK_AVAILABLE:
                wrapper = QATLoRAWrapper(model, use_lora=True, lora_rank=4)
                model = wrapper.enable_fp8_quantization()
                metadata["quantization_type"] = "FP8"
            elif quantization_type == "int4" and BENCHMARK_AVAILABLE:
                wrapper = QATLoRAWrapper(model, use_lora=True, lora_rank=4)
                model = wrapper.enable_int4_quantization()
                metadata["quantization_type"] = "INT4"
        elif quantization_type == "fp16":
            # FP16 is default in PyTorch when using CUDA
            metadata["quantization_type"] = "FP16"
        else:  # fp32
            metadata["quantization_type"] = "FP32"
            
        return model, metadata
    
    def create_test_data(self, batch_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create synthetic test data
        
        Returns:
            Tuple of (text_tokens, tabular_features, targets)
        """
        text_tokens = torch.randint(0, 1000, (batch_size, 32), device=self.device)
        tabular_features = torch.randn(batch_size, 20, device=self.device)
        targets = torch.randint(0, 2, (batch_size,), device=self.device)
        return text_tokens, tabular_features, targets
    
    def benchmark_forward_pass(self, model: torch.nn.Module, 
                             text_tokens: torch.Tensor, 
                             tabular_features: torch.Tensor,
                             num_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark forward pass performance
        
        Args:
            model: Model to benchmark
            text_tokens: Text input tokens
            tabular_features: Tabular input features
            num_iterations: Number of iterations to run
            
        Returns:
            Dictionary with benchmark results
        """
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(text_tokens, tabular_features)
                
        # Benchmark
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                outputs, aux_loss = model(text_tokens, tabular_features)
                
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = num_iterations / total_time
        
        return {
            "total_time": total_time,
            "avg_time_per_sample": avg_time,
            "samples_per_second": throughput,
            "iterations": num_iterations
        }
    
    def benchmark_memory_usage(self, model: torch.nn.Module,
                              text_tokens: torch.Tensor,
                              tabular_features: torch.Tensor) -> Dict[str, Any]:
        """Benchmark memory usage
        
        Args:
            model: Model to benchmark
            text_tokens: Text input tokens
            tabular_features: Tabular input features
            
        Returns:
            Dictionary with memory usage results
        """
        if self.device.type != 'cuda':
            return {"memory_usage_mb": 0, "memory_allocated_mb": 0}
            
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure memory before
        mem_before = torch.cuda.memory_allocated()
        
        # Run forward pass
        with torch.no_grad():
            outputs, aux_loss = model(text_tokens, tabular_features)
            
        # Measure memory after
        mem_after = torch.cuda.memory_allocated()
        mem_peak = torch.cuda.max_memory_allocated()
        
        return {
            "memory_before_mb": mem_before / (1024 * 1024),
            "memory_after_mb": mem_after / (1024 * 1024),
            "memory_peak_mb": mem_peak / (1024 * 1024),
            "memory_delta_mb": (mem_after - mem_before) / (1024 * 1024)
        }
    
    def benchmark_training_step(self, model: torch.nn.Module,
                               text_tokens: torch.Tensor,
                               tabular_features: torch.Tensor,
                               targets: torch.Tensor,
                               num_iterations: int = 50) -> Dict[str, Any]:
        """Benchmark training step performance
        
        Args:
            model: Model to benchmark
            text_tokens: Text input tokens
            tabular_features: Tabular input features
            targets: Target labels
            num_iterations: Number of iterations to run
            
        Returns:
            Dictionary with training benchmark results
        """
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Warmup
        for _ in range(3):
            optimizer.zero_grad()
            outputs, aux_loss = model(text_tokens, tabular_features)
            loss = criterion(outputs, targets)
            if aux_loss is not None:
                loss = loss + 0.01 * aux_loss
            loss.backward()
            optimizer.step()
            
        # Benchmark
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        for _ in range(num_iterations):
            optimizer.zero_grad()
            outputs, aux_loss = model(text_tokens, tabular_features)
            loss = criterion(outputs, targets)
            if aux_loss is not None:
                loss = loss + 0.01 * aux_loss
            loss.backward()
            optimizer.step()
            
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = num_iterations / total_time
        
        return {
            "total_time": total_time,
            "avg_time_per_step": avg_time,
            "steps_per_second": throughput,
            "iterations": num_iterations
        }
    
    def run_quantization_comparison(self, 
                                  quantization_types: list = ["fp32", "fp16", "fp8", "int4"],
                                  batch_size: int = 8,
                                  forward_iterations: int = 100,
                                  training_iterations: int = 50) -> Dict[str, Any]:
        """Run comprehensive quantization comparison benchmark
        
        Args:
            quantization_types: List of quantization types to compare
            batch_size: Batch size for testing
            forward_iterations: Number of forward pass iterations
            training_iterations: Number of training iterations
            
        Returns:
            Dictionary with comprehensive benchmark results
        """
        if not BENCHMARK_AVAILABLE:
            raise RuntimeError("MAHIA-X modules not available for benchmarking")
            
        print(f"🚀 Running quantization benchmark on {self.device}")
        print(f"   Quantization types: {quantization_types}")
        print(f"   Batch size: {batch_size}")
        print("-" * 60)
        
        results = {}
        
        for quant_type in quantization_types:
            print(f"\n📊 Benchmarking {quant_type.upper()}...")
            
            try:
                # Create model
                model, metadata = self.create_test_model(quant_type)
                print(f"   Model parameters: {metadata['model_params']:,}")
                print(f"   Model size: {metadata['model_size_mb']:.2f} MB")
                
                # Create test data
                text_tokens, tabular_features, targets = self.create_test_data(batch_size)
                
                # Run benchmarks
                forward_results = self.benchmark_forward_pass(
                    model, text_tokens, tabular_features, forward_iterations
                )
                
                memory_results = self.benchmark_memory_usage(
                    model, text_tokens, tabular_features
                )
                
                training_results = self.benchmark_training_step(
                    model, text_tokens, tabular_features, targets, training_iterations
                )
                
                # Store results
                results[quant_type] = {
                    "metadata": metadata,
                    "forward_benchmark": forward_results,
                    "memory_benchmark": memory_results,
                    "training_benchmark": training_results
                }
                
                # Print summary
                print(f"   Forward pass: {forward_results['avg_time_per_sample']*1000:.2f} ms/sample")
                print(f"   Throughput: {forward_results['samples_per_second']:.2f} samples/sec")
                if self.device.type == 'cuda':
                    print(f"   Peak memory: {memory_results['memory_peak_mb']:.2f} MB")
                print(f"   Training step: {training_results['avg_time_per_step']*1000:.2f} ms/step")
                
            except Exception as e:
                print(f"   ❌ Failed to benchmark {quant_type}: {e}")
                results[quant_type] = {"error": str(e)}
                
        # Store results
        self.results = results
        return results
    
    def print_comparison_report(self):
        """Print a formatted comparison report"""
        if not self.results:
            print("No benchmark results available")
            return
            
        print("\n" + "="*80)
        print("QUANTIZATION BENCHMARK COMPARISON REPORT")
        print("="*80)
        
        # Collect data for comparison
        quant_types = list(self.results.keys())
        forward_times = []
        throughputs = []
        memory_usage = []
        training_times = []
        
        for quant_type in quant_types:
            result = self.results[quant_type]
            if "error" in result:
                forward_times.append(float('inf'))
                throughputs.append(0)
                memory_usage.append(0)
                training_times.append(float('inf'))
            else:
                forward_times.append(result["forward_benchmark"]["avg_time_per_sample"] * 1000)
                throughputs.append(result["forward_benchmark"]["samples_per_second"])
                if self.device.type == 'cuda':
                    memory_usage.append(result["memory_benchmark"]["memory_peak_mb"])
                else:
                    memory_usage.append(0)
                training_times.append(result["training_benchmark"]["avg_time_per_step"] * 1000)
        
        # Find baseline (FP16 or FP32)
        baseline_idx = quant_types.index("fp16") if "fp16" in quant_types else quant_types.index("fp32") if "fp32" in quant_types else 0
        baseline_forward = forward_times[baseline_idx] if forward_times[baseline_idx] != float('inf') else 1.0
        baseline_throughput = throughputs[baseline_idx] if throughputs[baseline_idx] != 0 else 1.0
        baseline_memory = memory_usage[baseline_idx] if memory_usage[baseline_idx] != 0 else 1.0
        baseline_training = training_times[baseline_idx] if training_times[baseline_idx] != float('inf') else 1.0
        
        # Print comparison table
        print(f"{'Quantization':<12} {'Forward (ms)':<12} {'Throughput':<12} {'Memory (MB)':<12} {'Training (ms)':<12} {'Speedup':<10}")
        print("-" * 80)
        
        for i, quant_type in enumerate(quant_types):
            if forward_times[i] == float('inf'):
                speedup = "N/A"
            else:
                speedup = f"{baseline_forward / forward_times[i]:.2f}x"
                
            forward_str = f"{forward_times[i]:.2f}" if forward_times[i] != float('inf') else "ERROR"
            throughput_str = f"{throughputs[i]:.2f}" if throughputs[i] != 0 else "N/A"
            memory_str = f"{memory_usage[i]:.2f}" if memory_usage[i] != 0 else "N/A"
            training_str = f"{training_times[i]:.2f}" if training_times[i] != float('inf') else "ERROR"
            
            print(f"{quant_type.upper():<12} {forward_str:<12} {throughput_str:<12} {memory_str:<12} {training_str:<12} {speedup:<10}")
            
        print("="*80)
        
    def save_results(self, filepath: str = "./quantization_benchmark_results.pt"):
        """Save benchmark results to file
        
        Args:
            filepath: Path to save results (default: ./quantization_benchmark_results.pt)
        """
        if filepath is None:
            filepath = "./quantization_benchmark_results.pt"
            
        try:
            torch.save({
                "results": self.results,
                "device": str(self.device),
                "timestamp": time.time()
            }, filepath)
            print(f"✅ Benchmark results saved to {filepath}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


def main():
    """Main benchmark runner"""
    if not BENCHMARK_AVAILABLE:
        print("❌ MAHIA-X modules not available for benchmarking")
        return
        
    # Create benchmark runner
    runner = QuantizationBenchmarkRunner()
    
    # Run comparison benchmark
    results = runner.run_quantization_comparison(
        quantization_types=["fp32", "fp16", "fp8", "int4"],
        batch_size=8,
        forward_iterations=50,
        training_iterations=25
    )
    
    # Print comparison report
    runner.print_comparison_report()
    
    # Save results
    runner.save_results()
    
    print("\n✅ Quantization benchmark completed!")


if __name__ == '__main__':
    main()