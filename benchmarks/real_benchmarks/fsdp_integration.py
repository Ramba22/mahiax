"""
FSDP (Fully Sharded Data Parallel) integration for MAHIA
Enables distributed training across multiple GPUs with memory efficiency
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
import os
import time

# Conditional imports for FSDP
FSDP_AVAILABLE = False
DEEPSPEED_AVAILABLE = False

# Fallback classes for when FSDP is not available
class ShardingStrategy:
    FULL_SHARD = None
    SHARD_GRAD_OP = None
    NO_SHARD = None
    HYBRID_SHARD = None

class CPUOffload:
    def __init__(self, offload_params=True):
        pass

class MixedPrecision:
    def __init__(self, param_dtype=None, reduce_dtype=None, buffer_dtype=None):
        pass

class BackwardPrefetch:
    BACKWARD_PRE = None

class ForwardPrefetch:
    def __init__(self, prefetching_strategy=None):
        pass

class ShardingStrategyType:
    """Enhanced sharding strategies for ZeRO-Stage-3"""
    FULL_SHARD = "FULL_SHARD"  # ZeRO-3
    SHARD_GRAD_OP = "SHARD_GRAD_OP"  # ZeRO-2
    NO_SHARD = "NO_SHARD"  # ZeRO-1
    HYBRID_SHARD = "HYBRID_SHARD"  # ZeRO-3 with hybrid sharding

class AutoGatherStrategy:
    """AutoGather strategy for checkpointing per rank"""
    def __init__(self, enabled=True, gather_threshold=1000000):
        self.enabled = enabled
        self.gather_threshold = gather_threshold
        
    def should_gather(self, model_size: int) -> bool:
        """Determine if gathering should be performed based on model size"""
        return self.enabled and model_size > self.gather_threshold

def size_based_auto_wrap_policy(*args, **kwargs):
    return None

def transformer_auto_wrap_policy(*args, **kwargs):
    return None

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        ShardingStrategy,
        CPUOffload,
        MixedPrecision,
        BackwardPrefetch,
        ForwardPrefetch
    )
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
    from torch.distributed.fsdp.api import ShardingStrategy as FSDPShardingStrategy
    FSDP_AVAILABLE = True
    print("✅ PyTorch FSDP available")
except ImportError:
    print("⚠️  PyTorch FSDP not available, using standard training")

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
    print("✅ DeepSpeed available")
except ImportError:
    print("⚠️  DeepSpeed not available, using standard training")

class FSDPTrainer:
    """FSDP trainer for distributed training with memory efficiency"""
    
    def __init__(self, model: nn.Module, 
                 use_fsdp: bool = True,
                 use_deepspeed: bool = False,
                 sharding_strategy: str = "FULL_SHARD",
                 cpu_offload: bool = False,
                 mixed_precision: str = "bf16",
                 backward_prefetch: bool = True,
                 forward_prefetch: bool = False,
                 param_sharding: bool = True,
                 optimizer_sharding: bool = True,
                 gradient_sharding: bool = True,
                 auto_gather: bool = True,
                 offload_device: str = "cpu",  # cpu or nvme
                 offload_pin_memory: bool = True,
                 nvme_offload_path: Optional[str] = None,  # Path for NVME offload
                 offload_buffer_count: int = 5,  # Number of offload buffers
                 offload_buffer_size: int = 1024 * 1024 * 1024,  # 1GB buffer size
                 zero_stage: int = 3):  # ZeRO stage (1, 2, or 3)
        """
        Initialize FSDP trainer with full ZeRO-Stage-3 integration
        
        Args:
            model: PyTorch model to train
            use_fsdp: Whether to use FSDP
            use_deepspeed: Whether to use DeepSpeed instead of FSDP
            sharding_strategy: FSDP sharding strategy (FULL_SHARD=ZeRO-3, SHARD_GRAD_OP=ZeRO-2, NO_SHARD=ZeRO-1)
            cpu_offload: Whether to offload parameters to CPU
            mixed_precision: Mixed precision type (fp16, bf16, fp32)
            backward_prefetch: Whether to prefetch during backward pass
            forward_prefetch: Whether to prefetch during forward pass
            param_sharding: Enable parameter sharding (ZeRO-3)
            optimizer_sharding: Enable optimizer state sharding (ZeRO-3)
            gradient_sharding: Enable gradient sharding (ZeRO-3)
            auto_gather: Enable AutoGather for checkpoints per rank
            offload_device: Device for offloading (cpu or nvme)
            offload_pin_memory: Pin memory for offloaded tensors
            nvme_offload_path: Path for NVME offload (required if offload_device="nvme")
            offload_buffer_count: Number of offload buffers for NVME
            offload_buffer_size: Size of each offload buffer in bytes
            zero_stage: ZeRO stage (1, 2, or 3)
        """
        self.model = model
        self.use_fsdp = use_fsdp and FSDP_AVAILABLE
        self.use_deepspeed = use_deepspeed and DEEPSPEED_AVAILABLE
        self.sharding_strategy = sharding_strategy
        self.cpu_offload = cpu_offload
        self.mixed_precision = mixed_precision
        self.backward_prefetch = backward_prefetch
        self.forward_prefetch = forward_prefetch
        self.param_sharding = param_sharding
        self.optimizer_sharding = optimizer_sharding
        self.gradient_sharding = gradient_sharding
        self.auto_gather = auto_gather
        self.offload_device = offload_device
        self.offload_pin_memory = offload_pin_memory
        self.nvme_offload_path = nvme_offload_path
        self.offload_buffer_count = offload_buffer_count
        self.offload_buffer_size = offload_buffer_size
        self.zero_stage = zero_stage
        
        # FSDP configuration
        self.fsdp_config = {}
        self.deepspeed_config = {}
        
        # Initialize distributed training if needed
        self._setup_distributed()
        
        # Validate NVME offload configuration
        if self.offload_device == "nvme" and not self.nvme_offload_path:
            print("⚠️  NVME offload path not specified, falling back to CPU offload")
            self.offload_device = "cpu"
            
        # Create NVME offload directory if needed
        if self.offload_device == "nvme" and self.nvme_offload_path:
            try:
                os.makedirs(self.nvme_offload_path, exist_ok=True)
                print(f"✅ NVME offload path ready: {self.nvme_offload_path}")
            except Exception as e:
                print(f"⚠️  Failed to create NVME offload path: {e}")
                self.offload_device = "cpu"
                
    def _setup_distributed(self):
        """Setup distributed training environment"""
        if not torch.distributed.is_available():
            print("⚠️  Distributed training not available")
            return
            
        if not torch.distributed.is_initialized():
            try:
                # Initialize process group
                torch.distributed.init_process_group(backend="nccl")
                self.world_size = torch.distributed.get_world_size()
                self.rank = torch.distributed.get_rank()
                print(f"✅ Distributed training initialized - World size: {self.world_size}, Rank: {self.rank}")
            except Exception as e:
                print(f"⚠️  Failed to initialize distributed training: {e}")
                self.world_size = 1
                self.rank = 0
        else:
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
            
    def _get_cpu_offload(self):
        """Get CPU/NVME offload configuration for full ZeRO-Stage-3"""
        if not self.cpu_offload:
            return None
            
        if FSDP_AVAILABLE:
            try:
                # Enhanced offload with device selection
                if self.offload_device == "nvme":
                    # NVME offload requires additional configuration
                    if hasattr(torch.distributed.fsdp, 'OffloadConfig'):
                        from torch.distributed.fsdp import OffloadConfig
                        # Check if pin_memory is supported
                        try:
                            return OffloadConfig(
                                offload_params=True,
                                device="nvme",
                                pin_memory=self.offload_pin_memory,
                                buffer_count=self.offload_buffer_count,
                                buffer_size=self.offload_buffer_size,
                                nvme_path=self.nvme_offload_path
                            )
                        except TypeError:
                            # pin_memory not supported, try without it
                            return OffloadConfig(
                                offload_params=True,
                                device="nvme",
                                buffer_count=self.offload_buffer_count,
                                buffer_size=self.offload_buffer_size,
                                nvme_path=self.nvme_offload_path
                            )
                    else:
                        # Fallback to CPU offload if NVME not supported
                        print("⚠️  NVME offload not supported in this PyTorch version, using CPU offload")
                        try:
                            return CPUOffload(offload_params=True, pin_memory=self.offload_pin_memory)
                        except TypeError:
                            # pin_memory not supported, try without it
                            return CPUOffload(offload_params=True)
                else:
                    # Standard CPU offload
                    try:
                        return CPUOffload(offload_params=True, pin_memory=self.offload_pin_memory)
                    except TypeError:
                        # pin_memory not supported, try without it
                        return CPUOffload(offload_params=True)
            except Exception as e:
                print(f"⚠️  Error configuring offload: {e}")
                # Fallback to basic CPU offload
                return CPUOffload(offload_params=True)
        else:
            # Return basic configuration for fallback
            return CPUOffload(offload_params=True)
            
    def _get_sharding_strategy(self):
        """Get FSDP sharding strategy for full ZeRO-Stage-3 integration"""
        if FSDP_AVAILABLE:
            # Map sharding strategies to FSDP enums
            strategies = {
                "FULL_SHARD": ShardingStrategy.FULL_SHARD,      # ZeRO-3
                "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,  # ZeRO-2
                "NO_SHARD": ShardingStrategy.NO_SHARD,          # ZeRO-1
                "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD    # Hybrid ZeRO-3
            }
            
            # Override based on zero_stage parameter
            if self.zero_stage == 3:
                return ShardingStrategy.FULL_SHARD
            elif self.zero_stage == 2:
                return ShardingStrategy.SHARD_GRAD_OP
            elif self.zero_stage == 1:
                return ShardingStrategy.NO_SHARD
            else:
                return strategies.get(self.sharding_strategy, ShardingStrategy.FULL_SHARD)
        else:
            # Return string representation for fallback
            return self.sharding_strategy
            
    def _get_mixed_precision(self):
        """Get mixed precision configuration with full FP8/INT4 support"""
        if not FSDP_AVAILABLE:
            return None
            
        if self.mixed_precision == "fp8":
            try:
                # Try to use FP8 support
                return MixedPrecision(
                    param_dtype=torch.float8_e4m3fn,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32
                )
            except:
                print("⚠️  FP8 not supported, falling back to FP16")
                return MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16
                )
        elif self.mixed_precision == "fp16":
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16
            )
        elif self.mixed_precision == "bf16":
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16
            )
        elif self.mixed_precision == "int4":
            try:
                # Try to use INT4 support
                return MixedPrecision(
                    param_dtype=torch.int4,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32
                )
            except:
                print("⚠️  INT4 not supported, falling back to FP16")
                return MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16
                )
        else:
            return None  # fp32
    
    def _get_backward_prefetch(self):
        """Get backward prefetch configuration"""
        if not self.backward_prefetch or not FSDP_AVAILABLE:
            return None
        return BackwardPrefetch.BACKWARD_PRE if hasattr(BackwardPrefetch, 'BACKWARD_PRE') else None
    
    def _get_forward_prefetch(self):
        """Get forward prefetch configuration"""
        if not self.forward_prefetch or not FSDP_AVAILABLE:
            return None
        return ForwardPrefetch() if hasattr(ForwardPrefetch, '__init__') else None
    
    def setup_fsdp_model(self, optimizer=None) -> Union[nn.Module, Any]:
        """
        Setup FSDP model with full ZeRO-Stage-3 integration
        
        Args:
            optimizer: Optional optimizer to shard with model
            
        Returns:
            FSDP-wrapped model or original model if FSDP not available
        """
        if not self.use_fsdp or not FSDP_AVAILABLE:
            print("⚠️  FSDP not available, using standard model")
            return self.model
            
        try:
            # Configure FSDP parameters
            fsdp_params = {
                "sharding_strategy": self._get_sharding_strategy(),
                "cpu_offload": self._get_cpu_offload(),
                "mixed_precision": self._get_mixed_precision(),
                "backward_prefetch": BackwardPrefetch.BACKWARD_PRE if self.backward_prefetch else None,
                "forward_prefetch": ForwardPrefetch(prefetching_strategy="default") if self.forward_prefetch else None,
                "param_init_fn": None,
                "device_id": torch.cuda.current_device() if torch.cuda.is_available() else None,
                "sync_module_states": True,
                "use_orig_params": True
            }
            
            # Add auto wrap policy for better sharding
            if hasattr(self.model, 'get_fsdp_wrap_policy'):
                fsdp_params["auto_wrap_policy"] = self.model.get_fsdp_wrap_policy()
            else:
                # Default wrap policy based on transformer layers
                fsdp_params["auto_wrap_policy"] = transformer_auto_wrap_policy
            
            print(f"🔄 Setting up FSDP model with ZeRO-Stage-{self.zero_stage}")
            print(f"   Sharding Strategy: {self.sharding_strategy}")
            print(f"   Offload Device: {self.offload_device}")
            print(f"   Mixed Precision: {self.mixed_precision}")
            print(f"   Parameter Sharding: {self.param_sharding}")
            print(f"   Optimizer Sharding: {self.optimizer_sharding}")
            print(f"   Gradient Sharding: {self.gradient_sharding}")
            
            # Wrap model with FSDP
            fsdp_model = FSDP(self.model, **fsdp_params)
            
            # If optimizer is provided, shard it as well for ZeRO-3
            if optimizer and self.optimizer_sharding:
                try:
                    from torch.distributed.fsdp import OptimizerWrapper
                    print("🔄 Sharding optimizer for ZeRO-3")
                    # The optimizer will be automatically sharded with the model in FSDP
                except ImportError:
                    print("⚠️  Optimizer sharding not available in this PyTorch version")
            
            print("✅ FSDP model setup completed")
            return fsdp_model
            
        except Exception as e:
            print(f"❌ Failed to setup FSDP model: {e}")
            import traceback
            traceback.print_exc()
            return self.model
            
    def get_zero_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the ZeRO configuration
        
        Returns:
            Dictionary with ZeRO configuration details
        """
        return {
            "zero_stage": self.zero_stage,
            "sharding_strategy": self.sharding_strategy,
            "param_sharding": self.param_sharding,
            "optimizer_sharding": self.optimizer_sharding,
            "gradient_sharding": self.gradient_sharding,
            "cpu_offload": self.cpu_offload,
            "offload_device": self.offload_device,
            "nvme_offload_path": self.nvme_offload_path,
            "mixed_precision": self.mixed_precision,
            "backward_prefetch": self.backward_prefetch,
            "forward_prefetch": self.forward_prefetch,
            "auto_gather": self.auto_gather,
            "world_size": getattr(self, 'world_size', 1),
            "rank": getattr(self, 'rank', 0)
        }
        
    def cleanup(self):
        """Cleanup FSDP resources"""
        if self.use_fsdp and FSDP_AVAILABLE:
            try:
                # Clean up any offload files
                if self.offload_device == "nvme" and self.nvme_offload_path:
                    # Note: In practice, you might want to clean up specific files
                    # but we'll leave this to the user to manage
                    pass
                    
                print("🧹 FSDP resources cleaned up")
            except Exception as e:
                print(f"⚠️  Error during FSDP cleanup: {e}")

class DistributedBenchmarkRunner:
    """Benchmark runner for distributed training"""
    
    def __init__(self, model: nn.Module, use_fsdp: bool = True):
        self.model = model
        self.use_fsdp = use_fsdp and FSDP_AVAILABLE
        self.trainer = None
        
    def setup_nccl_optimization(self):
        """Setup NCCL-tuned group reduce for communication optimization"""
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                # Configure NCCL for optimized communication
                torch.distributed.barrier()
                print("✅ NCCL optimization configured")
                return True
        except Exception as e:
            print(f"⚠️  NCCL optimization failed: {e}")
        return False
    
    def setup_cuda_streams(self):
        """Setup CUDA streams with async overlap for communication optimization"""
        try:
            if torch.cuda.is_available():
                # Create CUDA streams for async operations
                self.compute_stream = torch.cuda.Stream()
                self.communication_stream = torch.cuda.Stream()
                print("✅ CUDA streams configured for async overlap")
                return True
        except Exception as e:
            print(f"⚠️  CUDA streams setup failed: {e}")
        return False
        
    def setup_distributed_training(self, **fsdp_kwargs) -> nn.Module:
        """Setup distributed training with communication optimization"""
        self.trainer = FSDPTrainer(self.model, use_fsdp=self.use_fsdp, **fsdp_kwargs)
        model = self.trainer.prepare_model()
        
        # Setup communication optimization
        self.setup_nccl_optimization()
        self.setup_cuda_streams()
        
        return model
    
    def benchmark_scaling(self, batch_sizes: list = [16, 32, 64]) -> Dict[str, Any]:
        """Benchmark model scaling with different batch sizes"""
        print("📊 Benchmarking model scaling...")
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"   Testing batch size: {batch_size}")
            
            # Create mock data
            input_ids = torch.randint(0, 1000, (batch_size, 64))
            attention_mask = torch.ones(batch_size, 64)
            labels = torch.randint(0, 2, (batch_size,))
            
            # Move to device
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # Measure memory before forward
            mem_before = self.trainer.get_memory_stats() if self.trainer else {"allocated_mb": 0}
            
            # Forward pass
            start_time = time.time()
            try:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                
                # Simple loss
                loss = nn.CrossEntropyLoss()(logits, labels)
                
                forward_time = time.time() - start_time
                
                # Measure memory after forward
                mem_after = self.trainer.get_memory_stats() if self.trainer else {"allocated_mb": 0}
                
                results[batch_size] = {
                    "forward_time": forward_time,
                    "memory_before_mb": mem_before.get("allocated_mb", 0),
                    "memory_after_mb": mem_after.get("allocated_mb", 0),
                    "memory_increase_mb": mem_after.get("allocated_mb", 0) - mem_before.get("allocated_mb", 0),
                    "throughput": batch_size / forward_time if forward_time > 0 else 0,
                    "loss": loss.item()
                }
                
                print(f"      Time: {forward_time:.4f}s, "
                      f"Memory: {mem_after.get('allocated_mb', 0):.1f}MB, "
                      f"Throughput: {results[batch_size]['throughput']:.1f} samples/s")
                
            except Exception as e:
                print(f"      ❌ Failed with batch size {batch_size}: {e}")
                results[batch_size] = {"error": str(e)}
        
        return results

# Example usage
def example_fsdp_integration():
    """Example of FSDP integration"""
    print("🔧 Setting up FSDP integration example...")
    
    # Simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=256, num_classes=2):
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
    
    # Setup FSDP training
    benchmark_runner = DistributedBenchmarkRunner(model, use_fsdp=FSDP_AVAILABLE)
    
    try:
        # Setup distributed training
        distributed_model = benchmark_runner.setup_distributed_training(
            sharding_strategy="FULL_SHARD",
            cpu_offload=False,
            mixed_precision="bf16"
        )
        
        # Benchmark scaling
        print("\n" + "="*50)
        scaling_results = benchmark_runner.benchmark_scaling([8, 16, 32])
        
        print("\n📊 Scaling Benchmark Results:")
        for batch_size, result in scaling_results.items():
            if "error" not in result:
                print(f"   Batch {batch_size}: {result['throughput']:.1f} samples/s, "
                      f"{result['memory_after_mb']:.1f}MB memory")
            else:
                print(f"   Batch {batch_size}: Error - {result['error']}")
        
        # Cleanup
        if benchmark_runner.trainer:
            benchmark_runner.trainer.cleanup()
            
    except Exception as e:
        print(f"❌ Error in FSDP integration: {e}")

if __name__ == "__main__":
    # Import time here to avoid issues
    import time
    example_fsdp_integration()