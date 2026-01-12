"""
Communication Optimizer for MAHIA
Implements NCCL-tuned group reduce, CUDA streams with async overlap and in-flight batching
"""

import torch
import time
from typing import Optional, Dict, Any, List
import threading

# Conditional imports for distributed training
NCCL_AVAILABLE = False
try:
    import torch.distributed as dist
    if dist.is_available():
        NCCL_AVAILABLE = True
        print("✅ NCCL distributed training available")
except ImportError:
    print("⚠️  NCCL distributed training not available")

CUDA_AVAILABLE = False
try:
    if torch.cuda.is_available():
        CUDA_AVAILABLE = True
        print("✅ CUDA available")
except ImportError:
    print("⚠️  CUDA not available")

class NCCLGroupReducer:
    """NCCL-tuned group reduce for optimized communication"""
    
    def __init__(self, group_size: int = 8, reduce_op: str = "sum"):
        self.group_size = group_size
        self.reduce_op = reduce_op
        self.process_group = None
        self.setup_process_group()
        
    def setup_process_group(self):
        """Setup NCCL process group for optimized communication"""
        if not NCCL_AVAILABLE:
            print("⚠️  NCCL not available, using fallback")
            return
            
        try:
            if dist.is_initialized():
                # Create process group with specific ranks
                world_size = dist.get_world_size()
                if world_size > self.group_size:
                    # Create sub-groups for more efficient communication
                    num_groups = world_size // self.group_size
                    for i in range(num_groups):
                        start_rank = i * self.group_size
                        end_rank = min((i + 1) * self.group_size, world_size)
                        ranks = list(range(start_rank, end_rank))
                        group = dist.new_group(ranks=ranks)
                        if dist.get_rank() in ranks:
                            self.process_group = group
                            print(f"✅ NCCL process group created with ranks {ranks}")
                            break
                else:
                    # Use default process group
                    self.process_group = dist.group.WORLD
                    print("✅ Using default NCCL process group")
            else:
                print("⚠️  Distributed training not initialized")
        except Exception as e:
            print(f"⚠️  NCCL process group setup failed: {e}")
    
    def group_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Perform group reduce operation with NCCL tuning"""
        if not NCCL_AVAILABLE or self.process_group is None:
            return tensor
            
        try:
            # Determine reduce operation
            if self.reduce_op == "sum":
                op = dist.ReduceOp.SUM
            elif self.reduce_op == "mean":
                op = dist.ReduceOp.AVG
            elif self.reduce_op == "min":
                op = dist.ReduceOp.MIN
            elif self.reduce_op == "max":
                op = dist.ReduceOp.MAX
            else:
                op = dist.ReduceOp.SUM
            
            # Perform all-reduce operation
            dist.all_reduce(tensor, op=op, group=self.process_group)
            
            # For mean operation, divide by world size
            if self.reduce_op == "mean":
                tensor = tensor / dist.get_world_size(self.process_group)
                
            return tensor
        except Exception as e:
            print(f"⚠️  NCCL group reduce failed: {e}")
            return tensor

class CUDAStreamManager:
    """Manage CUDA streams for async overlap"""
    
    def __init__(self, num_streams: int = 4):
        self.num_streams = num_streams
        self.compute_streams = []
        self.communication_streams = []
        self.current_stream_idx = 0
        self.setup_streams()
        
    def setup_streams(self):
        """Setup CUDA streams for async operations"""
        if not CUDA_AVAILABLE:
            print("⚠️  CUDA not available, async overlap disabled")
            return
            
        try:
            for i in range(self.num_streams):
                compute_stream = torch.cuda.Stream()
                comm_stream = torch.cuda.Stream()
                self.compute_streams.append(compute_stream)
                self.communication_streams.append(comm_stream)
            print(f"✅ Created {self.num_streams} CUDA stream pairs for async overlap")
        except Exception as e:
            print(f"⚠️  CUDA stream setup failed: {e}")
    
    def get_streams(self):
        """Get next stream pair for async operations"""
        if not self.compute_streams:
            return None, None
            
        compute_stream = self.compute_streams[self.current_stream_idx]
        comm_stream = self.communication_streams[self.current_stream_idx]
        
        # Rotate to next stream
        self.current_stream_idx = (self.current_stream_idx + 1) % self.num_streams
        
        return compute_stream, comm_stream
    
    def record_event(self, stream: torch.cuda.Stream) -> torch.cuda.Event:
        """Record event on stream for synchronization"""
        if stream is None:
            return None
        event = torch.cuda.Event()
        event.record(stream)
        return event
    
    def wait_event(self, stream: torch.cuda.Stream, event: torch.cuda.Event):
        """Wait for event on stream"""
        if stream is None or event is None:
            return
        stream.wait_event(event)

class InFlightBatchManager:
    """Manage in-flight batching for communication optimization"""
    
    def __init__(self, max_in_flight: int = 4):
        self.max_in_flight = max_in_flight
        self.in_flight_batches = []
        self.batch_counter = 0
        
    def submit_batch(self, batch_data: Any) -> int:
        """Submit batch for in-flight processing"""
        batch_id = self.batch_counter
        self.batch_counter += 1
        
        # Add to in-flight queue
        self.in_flight_batches.append({
            'id': batch_id,
            'data': batch_data,
            'submitted_time': time.time(),
            'completed': False
        })
        
        # Limit in-flight batches
        if len(self.in_flight_batches) > self.max_in_flight:
            # Remove oldest completed batches
            self.in_flight_batches = [
                b for b in self.in_flight_batches 
                if not b['completed']
            ][:self.max_in_flight]
        
        return batch_id
    
    def complete_batch(self, batch_id: int):
        """Mark batch as completed"""
        for batch in self.in_flight_batches:
            if batch['id'] == batch_id:
                batch['completed'] = True
                batch['completion_time'] = time.time()
                break
    
    def get_pending_batches(self) -> List[Dict]:
        """Get list of pending (not completed) batches"""
        return [b for b in self.in_flight_batches if not b['completed']]

class CommunicationOptimizer:
    """Main communication optimizer with NCCL, CUDA streams, and in-flight batching"""
    
    def __init__(self, 
                 group_size: int = 8,
                 num_streams: int = 4,
                 max_in_flight: int = 4,
                 reduce_op: str = "sum"):
        """
        Initialize communication optimizer
        
        Args:
            group_size: Size of NCCL communication groups
            num_streams: Number of CUDA streams for async overlap
            max_in_flight: Maximum number of in-flight batches
            reduce_op: Reduction operation for NCCL (sum, mean, min, max)
        """
        self.nccl_reducer = NCCLGroupReducer(group_size=group_size, reduce_op=reduce_op)
        self.stream_manager = CUDAStreamManager(num_streams=num_streams)
        self.batch_manager = InFlightBatchManager(max_in_flight=max_in_flight)
        
        # Performance tracking
        self.stats = {
            'reductions': 0,
            'async_overlaps': 0,
            'in_flight_batches': 0,
            'total_comm_time': 0.0
        }
        
        print("✅ Communication Optimizer initialized")
        print(f"   Group Size: {group_size}")
        print(f"   Streams: {num_streams}")
        print(f"   In-Flight Batches: {max_in_flight}")
        print(f"   Reduce Op: {reduce_op}")
    
    def optimized_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Perform optimized reduction with NCCL tuning"""
        start_time = time.time()
        
        # Perform NCCL group reduce
        result = self.nccl_reducer.group_reduce(tensor)
        
        # Update stats
        self.stats['reductions'] += 1
        self.stats['total_comm_time'] += (time.time() - start_time)
        
        return result
    
    def async_compute_communicate(self, compute_fn, comm_fn):
        """
        Perform async compute and communication overlap
        
        Args:
            compute_fn: Function to perform compute operation
            comm_fn: Function to perform communication operation
            
        Returns:
            Result of compute operation
        """
        start_time = time.time()
        
        # Get streams for async overlap
        compute_stream, comm_stream = self.stream_manager.get_streams()
        
        if compute_stream is None or comm_stream is None:
            # Fallback to sequential execution
            compute_result = compute_fn()
            comm_result = comm_fn()
            self.stats['total_comm_time'] += (time.time() - start_time)
            return compute_result
        
        try:
            # Perform compute on compute stream
            with torch.cuda.stream(compute_stream):
                compute_result = compute_fn()
            
            # Perform communication on communication stream
            with torch.cuda.stream(comm_stream):
                comm_result = comm_fn()
            
            # Wait for both streams to complete
            torch.cuda.synchronize()
            
            # Update stats
            self.stats['async_overlaps'] += 1
            self.stats['total_comm_time'] += (time.time() - start_time)
            
            return compute_result
            
        except Exception as e:
            print(f"⚠️  Async overlap failed: {e}")
            # Fallback to sequential execution
            compute_result = compute_fn()
            comm_result = comm_fn()
            self.stats['total_comm_time'] += (time.time() - start_time)
            return compute_result
    
    def submit_in_flight_batch(self, batch_data: Any) -> int:
        """Submit batch for in-flight processing"""
        batch_id = self.batch_manager.submit_batch(batch_data)
        self.stats['in_flight_batches'] += 1
        return batch_id
    
    def complete_in_flight_batch(self, batch_id: int):
        """Mark in-flight batch as completed"""
        self.batch_manager.complete_batch(batch_id)
    
    def get_pending_batches(self) -> List[Dict]:
        """Get list of pending in-flight batches"""
        return self.batch_manager.get_pending_batches()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get communication optimization statistics"""
        return self.stats.copy()
    
    def print_stats(self):
        """Print communication optimization statistics"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("COMMUNICATION OPTIMIZER STATISTICS")
        print("="*50)
        print(f"Reductions: {stats['reductions']}")
        print(f"Async Overlaps: {stats['async_overlaps']}")
        print(f"In-Flight Batches: {stats['in_flight_batches']}")
        print(f"Total Communication Time: {stats['total_comm_time']:.4f}s")
        if stats['reductions'] > 0:
            avg_time = stats['total_comm_time'] / stats['reductions']
            print(f"Average Time per Reduction: {avg_time*1000:.2f}ms")
        print("="*50)

# Example usage
def example_communication_optimization():
    """Example of communication optimization usage"""
    print("🔧 Setting up Communication Optimizer example...")
    
    # Create optimizer
    optimizer = CommunicationOptimizer(
        group_size=4,
        num_streams=2,
        max_in_flight=2,
        reduce_op="sum"
    )
    
    # Simulate tensor operations
    if CUDA_AVAILABLE and torch.cuda.is_available():
        device = torch.device("cuda")
        tensor1 = torch.randn(1000, 1000, device=device)
        tensor2 = torch.randn(1000, 1000, device=device)
        
        print("\n🚀 Testing optimized reduction...")
        start_time = time.time()
        result = optimizer.optimized_reduce(tensor1.clone())
        reduce_time = time.time() - start_time
        print(f"   Reduction completed in {reduce_time*1000:.2f}ms")
        
        print("\n🚀 Testing async compute-communicate overlap...")
        def compute_fn():
            # Simulate compute operation
            return torch.matmul(tensor1, tensor2)
        
        def comm_fn():
            # Simulate communication operation
            return optimizer.optimized_reduce(tensor1.clone())
        
        start_time = time.time()
        compute_result = optimizer.async_compute_communicate(compute_fn, comm_fn)
        async_time = time.time() - start_time
        print(f"   Async overlap completed in {async_time*1000:.2f}ms")
        
        print("\n🚀 Testing in-flight batching...")
        batch_ids = []
        for i in range(3):
            batch_id = optimizer.submit_in_flight_batch(f"batch_{i}")
            batch_ids.append(batch_id)
            print(f"   Submitted batch {batch_id}")
        
        # Complete some batches
        for batch_id in batch_ids[:2]:
            optimizer.complete_in_flight_batch(batch_id)
            print(f"   Completed batch {batch_id}")
        
        # Show pending batches
        pending = optimizer.get_pending_batches()
        print(f"   Pending batches: {len(pending)}")
        
        # Print statistics
        optimizer.print_stats()
    else:
        print("⚠️  CUDA not available, showing configuration only")
        optimizer.print_stats()

if __name__ == "__main__":
    example_communication_optimization()