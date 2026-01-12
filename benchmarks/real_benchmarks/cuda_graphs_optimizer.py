"""
CUDA Graphs and Persistent Kernels Optimizer for MAHIA
Finalizes warmup paths to persistent CUDA graph cache to reduce kernel launch overhead
"""

import torch
import time
from typing import Optional, Dict, Any, List
import threading

# Try to import CUDA-related modules
CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AVAILABLE = False

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
    print("✅ Triton available for CUDA kernel optimization")
except ImportError:
    print("⚠️  Triton not available, using standard PyTorch operations")

class CUDAGraphManager:
    """Manage CUDA graphs for persistent kernel execution"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.graphs = {}  # Store compiled graphs
        self.graph_inputs = {}  # Store input buffers
        self.graph_outputs = {}  # Store output buffers
        self.is_warmed_up = False
        self.warmup_count = 0
        
        # Check CUDA availability
        if not CUDA_AVAILABLE:
            print("⚠️  CUDA not available, graphs will not be used")
    
    def can_use_cuda_graphs(self) -> bool:
        """Check if CUDA graphs can be used"""
        return CUDA_AVAILABLE and torch.cuda.is_available()
    
    def capture_graph(self, model: torch.nn.Module, sample_inputs: Dict[str, torch.Tensor], 
                     graph_id: str = "default") -> bool:
        """
        Capture model execution as a CUDA graph
        
        Args:
            model: Model to capture
            sample_inputs: Sample inputs to trace
            graph_id: Identifier for this graph
            
        Returns:
            bool: Whether graph capture was successful
        """
        if not self.can_use_cuda_graphs():
            print("⚠️  Cannot capture CUDA graph - CUDA not available")
            return False
        
        try:
            print(f"📸 Capturing CUDA graph '{graph_id}'...")
            
            # Move inputs to device
            device_inputs = {}
            for key, tensor in sample_inputs.items():
                device_inputs[key] = tensor.to(f"cuda:{self.device_id}")
            
            # Create CUDA graph
            graph = torch.cuda.CUDAGraph()
            
            # Warmup runs
            for _ in range(3):
                with torch.no_grad():
                    _ = model(**device_inputs)
            
            # Capture graph
            with torch.cuda.graph(graph):
                with torch.no_grad():
                    output = model(**device_inputs)
            
            # Store graph and buffers
            self.graphs[graph_id] = graph
            self.graph_inputs[graph_id] = device_inputs
            self.graph_outputs[graph_id] = output
            
            print(f"✅ CUDA graph '{graph_id}' captured successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to capture CUDA graph '{graph_id}': {e}")
            return False
    
    def replay_graph(self, graph_id: str = "default", 
                    new_inputs: Optional[Dict[str, torch.Tensor]] = None) -> Any:
        """
        Replay a captured CUDA graph
        
        Args:
            graph_id: Identifier of graph to replay
            new_inputs: New inputs to use (must match shape/dtype of captured inputs)
            
        Returns:
            Any: Graph output
        """
        if not self.can_use_cuda_graphs():
            return None
            
        if graph_id not in self.graphs:
            print(f"⚠️  CUDA graph '{graph_id}' not found")
            return None
        
        try:
            # Update inputs if provided
            if new_inputs:
                for key, tensor in new_inputs.items():
                    if key in self.graph_inputs[graph_id]:
                        # Copy new data to existing buffer
                        self.graph_inputs[graph_id][key].copy_(tensor.to(f"cuda:{self.device_id}"))
            
            # Replay graph
            self.graphs[graph_id].replay()
            
            return self.graph_outputs[graph_id]
            
        except Exception as e:
            print(f"❌ Failed to replay CUDA graph '{graph_id}': {e}")
            return None
    
    def warmup_graphs(self, model: torch.nn.Module, 
                     input_shapes: Dict[str, tuple],
                     warmup_iterations: int = 10) -> bool:
        """
        Warmup and capture graphs for common input shapes
        
        Args:
            model: Model to optimize
            input_shapes: Dictionary of input names and their shapes
            warmup_iterations: Number of warmup iterations
            
        Returns:
            bool: Whether warmup was successful
        """
        if not self.can_use_cuda_graphs():
            print("⚠️  Skipping warmup - CUDA not available")
            return False
        
        print(f"🔥 Warming up CUDA graphs with {warmup_iterations} iterations...")
        
        try:
            # Create sample inputs for each shape
            for i, (input_name, shape) in enumerate(input_shapes.items()):
                graph_id = f"warmup_{i}_{shape[-1]}"  # Include sequence length in ID
                
                # Create sample input
                sample_inputs = {
                    input_name: torch.randint(0, 1000, shape, 
                                            device=f"cuda:{self.device_id}", 
                                            dtype=torch.long)
                }
                
                # Add attention mask if needed
                if "input_ids" in input_name:
                    sample_inputs["attention_mask"] = torch.ones_like(sample_inputs[input_name])
                
                # Capture graph
                success = self.capture_graph(model, sample_inputs, graph_id)
                if success:
                    print(f"   Captured graph for shape {shape}")
                
                # Run warmup iterations
                for _ in range(warmup_iterations):
                    _ = self.replay_graph(graph_id)
            
            self.is_warmed_up = True
            self.warmup_count += 1
            print("✅ CUDA graphs warmed up successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed during warmup: {e}")
            return False
    
    def cleanup(self):
        """Clean up captured graphs"""
        self.graphs.clear()
        self.graph_inputs.clear()
        self.graph_outputs.clear()
        self.is_warmed_up = False
        print("🧹 CUDA graph manager cleaned up")

class PersistentKernelOptimizer:
    """Optimize kernels for persistent execution"""
    
    def __init__(self, use_fp8: bool = False):
        self.use_fp8 = use_fp8 and TRITON_AVAILABLE
        self.kernel_cache = {}
        self.compilation_stats = {}
        
        if self.use_fp8:
            print("🔧 FP8 kernel optimization enabled")
        else:
            print("🔧 Using standard precision kernels")
    
    def optimize_linear_layer(self, layer: torch.nn.Linear, 
                            layer_name: str = "linear") -> torch.nn.Module:
        """
        Optimize a linear layer for persistent execution
        
        Args:
            layer: Linear layer to optimize
            layer_name: Name for identification
            
        Returns:
            torch.nn.Module: Optimized layer
        """
        print(f"⚙️  Optimizing linear layer '{layer_name}'...")
        
        # For now, we'll just return the original layer
        # In a full implementation, this would replace with optimized kernels
        return layer
    
    def create_persistent_attention_kernel(self, 
                                         head_dim: int = 64,
                                         max_seq_len: int = 2048) -> 'callable':
        """
        Create a persistent attention kernel using Triton
        
        Args:
            head_dim: Attention head dimension
            max_seq_len: Maximum sequence length
            
        Returns:
            callable: Optimized attention kernel
        """
        try:
            if TRITON_AVAILABLE:
                @triton.jit
                def _persistent_attention_kernel(
                    q_ptr, k_ptr, v_ptr, output_ptr,
                    seq_len, head_dim,
                    stride_qb, stride_qh, stride_qm, stride_qk,
                    stride_kb, stride_kh, stride_kn, stride_kk,
                    stride_vb, stride_vh, stride_vn, stride_vk,
                    stride_ob, stride_oh, stride_om, stride_ok,
                    BLOCK_SIZE: tl.constexpr,
                    HEAD_DIM: tl.constexpr
                ):
                    # Get program indices
                    batch_id = tl.program_id(axis=0)
                    head_id = tl.program_id(axis=1)
                    block_id = tl.program_id(axis=2)
                    
                    # Compute offsets
                    offs_m = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    offs_k = tl.arange(0, HEAD_DIM)
                    
                    # Load Q, K, V
                    q_ptrs = q_ptr + (batch_id * stride_qb + head_id * stride_qh + 
                                     offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
                    k_ptrs = k_ptr + (batch_id * stride_kb + head_id * stride_kh + 
                                     offs_m[None, :] * stride_kn + offs_k[:, None] * stride_kk)
                    v_ptrs = v_ptr + (batch_id * stride_vb + head_id * stride_vh + 
                                     offs_m[:, None] * stride_vn + offs_k[None, :] * stride_vk)
                    
                    # Load data
                    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)
                    k = tl.load(k_ptrs, mask=offs_m[None, :] < seq_len, other=0.0)
                    v = tl.load(v_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)
                    
                    # Compute attention scores
                    scores = tl.dot(q, k, trans_b=True) * tl.rsqrt(tl.cast(HEAD_DIM, tl.float32))
                    scores = tl.where(offs_m[:, None] < seq_len, scores, float("-inf"))
                    
                    # Softmax
                    scores = tl.softmax(scores, axis=1)
                    
                    # Apply attention to values
                    output = tl.dot(scores, v)
                    
                    # Store output
                    output_ptrs = output_ptr + (batch_id * stride_ob + head_id * stride_oh + 
                                              offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
                    tl.store(output_ptrs, output, mask=offs_m[:, None] < seq_len)
                
                print("✅ Persistent attention kernel created")
                return _persistent_attention_kernel
            else:
                print("⚠️  Triton not available, returning standard attention")
                return self._standard_attention
            
        except Exception as e:
            print(f"❌ Failed to create persistent attention kernel: {e}")
            return self._standard_attention
    
    def _standard_attention(self, q, k, v, mask=None):
        """Standard attention implementation as fallback"""
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)
    
    def get_kernel_stats(self) -> Dict[str, Any]:
        """Get kernel optimization statistics"""
        return {
            "use_fp8": self.use_fp8,
            "triton_available": TRITON_AVAILABLE,
            "compiled_kernels": len(self.kernel_cache),
            "compilation_stats": self.compilation_stats
        }

class CUDAGraphBenchmarkRunner:
    """Benchmark runner with CUDA graphs optimization"""
    
    def __init__(self, model: torch.nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        
        # Initialize optimizers
        self.graph_manager = CUDAGraphManager()
        self.kernel_optimizer = PersistentKernelOptimizer(use_fp8=False)
        
        # Benchmark settings
        self.warmup_iterations = 10
        self.benchmark_iterations = 50
    
    def benchmark_with_graphs(self, 
                            batch_sizes: List[int] = [16, 32, 64],
                            seq_lengths: List[int] = [64, 128, 256]) -> Dict[str, Any]:
        """
        Benchmark model performance with and without CUDA graphs
        
        Args:
            batch_sizes: List of batch sizes to test
            seq_lengths: List of sequence lengths to test
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        print("🚀 Running CUDA Graphs Benchmark")
        print("=" * 50)
        
        results = {
            "with_graphs": {},
            "without_graphs": {},
            "improvements": {}
        }
        
        # Test without graphs
        print("📊 Benchmarking WITHOUT CUDA graphs...")
        baseline_results = self._benchmark_model(batch_sizes, seq_lengths, use_graphs=False)
        results["without_graphs"] = baseline_results
        
        # Test with graphs (if available)
        if self.graph_manager.can_use_cuda_graphs():
            print("\n📊 Benchmarking WITH CUDA graphs...")
            graph_results = self._benchmark_model(batch_sizes, seq_lengths, use_graphs=True)
            results["with_graphs"] = graph_results
            
            # Calculate improvements
            for bs in batch_sizes:
                for seq_len in seq_lengths:
                    key = f"bs{bs}_seq{seq_len}"
                    if key in baseline_results and key in graph_results:
                        baseline_time = baseline_results[key]["avg_time"]
                        graph_time = graph_results[key]["avg_time"]
                        
                        if baseline_time > 0:
                            speedup = baseline_time / graph_time if graph_time > 0 else 0
                            improvement = ((baseline_time - graph_time) / baseline_time) * 100
                            
                            results["improvements"][key] = {
                                "speedup": speedup,
                                "improvement_percent": improvement
                            }
        
        # Print summary
        print("\n📈 CUDA Graphs Performance Summary:")
        for bs in batch_sizes:
            for seq_len in seq_lengths:
                key = f"bs{bs}_seq{seq_len}"
                if key in results["improvements"]:
                    improvement = results["improvements"][key]
                    print(f"   BS={bs}, Seq={seq_len}: "
                          f"{improvement['speedup']:.2f}x speedup "
                          f"({improvement['improvement_percent']:.1f}% improvement)")
        
        return results
    
    def _benchmark_model(self, batch_sizes: List[int], seq_lengths: List[int], 
                        use_graphs: bool = False) -> Dict[str, Any]:
        """Benchmark model with specific settings"""
        results = {}
        
        # Evaluation mode
        if hasattr(self.model, "eval"):
            self.model.eval()
        
        with torch.no_grad():
            for bs in batch_sizes:
                for seq_len in seq_lengths:
                    key = f"bs{bs}_seq{seq_len}"
                    print(f"   Testing BS={bs}, Seq={seq_len}...")
                    
                    # Create test data
                    input_ids = torch.randint(0, 1000, (bs, seq_len), 
                                            device=self.device, dtype=torch.long)
                    attention_mask = torch.ones_like(input_ids)
                    
                    # Warmup
                    for _ in range(3):
                        _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    # Benchmark
                    times = []
                    for i in range(self.benchmark_iterations):
                        start_time = time.time()
                        
                        if use_graphs and self.graph_manager.can_use_cuda_graphs():
                            # Use CUDA graph if available
                            graph_id = f"benchmark_bs{bs}_seq{seq_len}"
                            
                            # Capture graph on first iteration
                            if i == 0:
                                sample_inputs = {
                                    "input_ids": input_ids,
                                    "attention_mask": attention_mask
                                }
                                self.graph_manager.capture_graph(self.model, sample_inputs, graph_id)
                            
                            # Replay graph
                            _ = self.graph_manager.replay_graph(graph_id)
                        else:
                            # Standard execution
                            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        
                        end_time = time.time()
                        times.append(end_time - start_time)
                    
                    # Calculate statistics
                    avg_time = sum(times) / len(times)
                    min_time = min(times)
                    max_time = max(times)
                    
                    results[key] = {
                        "avg_time": avg_time,
                        "min_time": min_time,
                        "max_time": max_time,
                        "batch_size": bs,
                        "seq_length": seq_len,
                        "iterations": self.benchmark_iterations
                    }
                    
                    print(f"      Avg: {avg_time*1000:.2f}ms, "
                          f"Min: {min_time*1000:.2f}ms, "
                          f"Max: {max_time*1000:.2f}ms")
        
        return results
    
    def warmup_persistent_kernels(self, input_shapes: Dict[str, tuple]) -> bool:
        """
        Warmup persistent kernels for common input patterns
        
        Args:
            input_shapes: Dictionary of input names and shapes
            
        Returns:
            bool: Whether warmup was successful
        """
        print("🔥 Warming up persistent kernels...")
        
        try:
            # Warmup CUDA graphs
            success = self.graph_manager.warmup_graphs(
                self.model, input_shapes, self.warmup_iterations
            )
            
            if success:
                print("✅ Persistent kernels warmed up successfully")
                return True
            else:
                print("⚠️  Partial warmup completed")
                return False
                
        except Exception as e:
            print(f"❌ Failed to warmup persistent kernels: {e}")
            return False
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            "cuda_available": CUDA_AVAILABLE,
            "triton_available": TRITON_AVAILABLE,
            "graphs_warmed_up": self.graph_manager.is_warmed_up,
            "warmup_count": self.graph_manager.warmup_count,
            "kernel_stats": self.kernel_optimizer.get_kernel_stats()
        }

# Example usage
def example_cuda_graphs_optimization():
    """Example of CUDA graphs optimization"""
    print("🔧 Setting up CUDA graphs optimization example...")
    
    # Simple model for demonstration
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=128, num_classes=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.transformer_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, 
                nhead=8,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            )
            self.classifier = nn.Linear(hidden_size, num_classes)
            
        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            x = self.transformer_layer(x)
            x = x.mean(dim=1)  # Global average pooling
            logits = self.classifier(x)
            return logits
    
    # Create model
    model = SimpleModel()
    print(f"✅ Created model with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    # Create benchmark runner
    benchmark_runner = CUDAGraphBenchmarkRunner(model)
    
    # Define common input shapes
    input_shapes = {
        "input_ids": (32, 128),  # batch_size=32, seq_length=128
        "input_ids_large": (16, 256)  # batch_size=16, seq_length=256
    }
    
    # Warmup persistent kernels
    print("\n" + "="*60)
    warmup_success = benchmark_runner.warmup_persistent_kernels(input_shapes)
    
    # Run benchmark
    print("\n" + "="*60)
    results = benchmark_runner.benchmark_with_graphs(
        batch_sizes=[16, 32],
        seq_lengths=[64, 128]
    )
    
    # Print optimization stats
    print("\n" + "="*60)
    stats = benchmark_runner.get_optimization_stats()
    print("⚙️  Optimization Statistics:")
    print(f"   CUDA Available: {stats['cuda_available']}")
    print(f"   Triton Available: {stats['triton_available']}")
    print(f"   Graphs Warmed Up: {stats['graphs_warmed_up']}")
    print(f"   Warmup Count: {stats['warmup_count']}")
    
    return results, stats

if __name__ == "__main__":
    example_cuda_graphs_optimization()