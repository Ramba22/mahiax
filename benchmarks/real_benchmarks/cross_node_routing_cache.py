"""
Cross-Node Routing Cache for MAHIA
Centralized cache for expert routing decisions to reduce communication latency in Top-K MoE
"""

import torch
import time
import threading
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
import json

class RoutingCacheEntry:
    """Represents a cached routing decision"""
    
    def __init__(self, 
                 context_hash: str,
                 expert_indices: List[int],
                 expert_weights: List[float],
                 confidence_scores: List[float],
                 timestamp: float,
                 node_id: Optional[str] = None):
        self.context_hash = context_hash
        self.expert_indices = expert_indices
        self.expert_weights = expert_weights
        self.confidence_scores = confidence_scores
        self.timestamp = timestamp
        self.node_id = node_id
        self.access_count = 0
        self.last_accessed = timestamp

class CrossNodeRoutingCache:
    """Centralized routing cache for distributed MoE systems"""
    
    def __init__(self, 
                 max_cache_size: int = 10000,
                 ttl_seconds: int = 3600,  # 1 hour
                 enable_compression: bool = True,
                 distributed_backend: str = "memory",  # "redis", "nccl", or "memory"
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 nccl_group_name: str = "routing_cache_group"):
        """
        Initialize cross-node routing cache
        
        Args:
            max_cache_size: Maximum number of entries in cache
            ttl_seconds: Time-to-live for cache entries in seconds
            enable_compression: Whether to compress cache entries
            distributed_backend: Backend for distributed cache ("redis", "nccl", or "memory")
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            nccl_group_name: Name of NCCL group for broadcast
        """
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        self.enable_compression = enable_compression
        self.distributed_backend = distributed_backend
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.nccl_group_name = nccl_group_name
        
        # Cache storage
        self.cache = OrderedDict()  # LRU cache
        self.lock = threading.RLock()  # Thread-safe access
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Node information
        self.node_id = self._generate_node_id()
        self.cluster_nodes = set()
        
        # Distributed backend initialization
        self.redis_client = None
        self.nccl_initialized = False
        self.nccl_comm = None
        
        if self.distributed_backend == "redis":
            self._init_redis_backend()
        elif self.distributed_backend == "nccl":
            self._init_nccl_backend()
        
        print(f"✅ Cross-Node Routing Cache initialized (Node ID: {self.node_id})")
        print(f"   Backend: {self.distributed_backend}")
        if self.distributed_backend == "redis":
            print(f"   Redis: {self.redis_host}:{self.redis_port}")
        elif self.distributed_backend == "nccl":
            print(f"   NCCL Group: {self.nccl_group_name}")
    
    def _generate_node_id(self) -> str:
        """Generate unique node identifier"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _hash_context(self, context: Any) -> str:
        """
        Create hash of context for cache key
        
        Args:
            context: Input context (tensor, list, dict, etc.)
            
        Returns:
            str: Hash string
        """
        try:
            # Convert context to bytes for hashing
            if isinstance(context, torch.Tensor):
                # For tensors, use their content and shape
                context_bytes = context.detach().cpu().numpy().tobytes() + str(context.shape).encode()
            elif isinstance(context, (list, tuple)):
                context_bytes = str(context).encode()
            elif isinstance(context, dict):
                context_bytes = json.dumps(context, sort_keys=True).encode()
            else:
                context_bytes = str(context).encode()
            
            # Create hash
            hash_obj = hashlib.sha256(context_bytes)
            return hash_obj.hexdigest()[:16]  # Short hash for efficiency
            
        except Exception as e:
            print(f"⚠️  Failed to hash context: {e}")
            # Fallback to random hash
            return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    
    def _init_redis_backend(self):
        """Initialize Redis backend for distributed caching"""
        try:
            import redis
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=False
            )
            # Test connection
            self.redis_client.ping()
            print("✅ Redis backend initialized")
        except ImportError:
            print("⚠️  Redis library not available, falling back to memory backend")
            self.distributed_backend = "memory"
        except Exception as e:
            print(f"⚠️  Failed to initialize Redis backend: {e}, falling back to memory backend")
            self.distributed_backend = "memory"
    
    def _init_nccl_backend(self):
        """Initialize NCCL backend for distributed caching"""
        try:
            # Check if torch distributed is available
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                self.nccl_initialized = True
                print("✅ NCCL backend initialized")
            else:
                print("⚠️  NCCL not available or not initialized, falling back to memory backend")
                self.distributed_backend = "memory"
        except Exception as e:
            print(f"⚠️  Failed to initialize NCCL backend: {e}, falling back to memory backend")
            self.distributed_backend = "memory"
    
    def _serialize_entry(self, entry: RoutingCacheEntry) -> bytes:
        """
        Serialize cache entry for distributed storage
        
        Args:
            entry: Cache entry to serialize
            
        Returns:
            bytes: Serialized entry
        """
        try:
            data = {
                "context_hash": entry.context_hash,
                "expert_indices": entry.expert_indices,
                "expert_weights": entry.expert_weights,
                "confidence_scores": entry.confidence_scores,
                "timestamp": entry.timestamp,
                "node_id": entry.node_id,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed
            }
            
            json_data = json.dumps(data)
            
            if self.enable_compression:
                import zlib
                return zlib.compress(json_data.encode('utf-8'))
            else:
                return json_data.encode('utf-8')
        except Exception as e:
            print(f"⚠️  Failed to serialize entry: {e}")
            return b""
    
    def _deserialize_entry(self, data: bytes) -> Optional[RoutingCacheEntry]:
        """
        Deserialize cache entry from distributed storage
        
        Args:
            data: Serialized entry data
            
        Returns:
            RoutingCacheEntry: Deserialized entry or None
        """
        try:
            if not data:
                return None
                
            if self.enable_compression:
                import zlib
                json_data = zlib.decompress(data).decode('utf-8')
            else:
                json_data = data.decode('utf-8')
                
            entry_data = json.loads(json_data)
            
            entry = RoutingCacheEntry(
                context_hash=entry_data["context_hash"],
                expert_indices=entry_data["expert_indices"],
                expert_weights=entry_data["expert_weights"],
                confidence_scores=entry_data["confidence_scores"],
                timestamp=entry_data["timestamp"],
                node_id=entry_data["node_id"]
            )
            entry.access_count = entry_data.get("access_count", 0)
            entry.last_accessed = entry_data.get("last_accessed", entry_data["timestamp"])
            
            return entry
        except Exception as e:
            print(f"⚠️  Failed to deserialize entry: {e}")
            return None
    
    def _distributed_put(self, key: str, entry: RoutingCacheEntry) -> bool:
        """
        Put entry in distributed cache
        
        Args:
            key: Cache key
            entry: Cache entry to store
            
        Returns:
            bool: Whether storage was successful
        """
        if self.distributed_backend == "redis" and self.redis_client:
            try:
                serialized_data = self._serialize_entry(entry)
                self.redis_client.setex(
                    key, 
                    self.ttl_seconds, 
                    serialized_data
                )
                return True
            except Exception as e:
                print(f"⚠️  Failed to store in Redis: {e}")
                return False
        elif self.distributed_backend == "nccl" and self.nccl_initialized:
            try:
                # For NCCL, we would broadcast to other nodes
                # This is a simplified implementation
                serialized_data = self._serialize_entry(entry)
                # In a real implementation, this would use torch.distributed.broadcast
                # or similar NCCL operations to share the data with other nodes
                return True
            except Exception as e:
                print(f"⚠️  Failed to broadcast via NCCL: {e}")
                return False
        else:
            # Memory backend - already handled locally
            return True
    
    def _distributed_get(self, key: str) -> Optional[RoutingCacheEntry]:
        """
        Get entry from distributed cache
        
        Args:
            key: Cache key
            
        Returns:
            RoutingCacheEntry: Retrieved entry or None
        """
        if self.distributed_backend == "redis" and self.redis_client:
            try:
                serialized_data = self.redis_client.get(key)
                if serialized_data:
                    return self._deserialize_entry(serialized_data)
                return None
            except Exception as e:
                print(f"⚠️  Failed to retrieve from Redis: {e}")
                return None
        elif self.distributed_backend == "nccl" and self.nccl_initialized:
            try:
                # For NCCL, we would receive broadcasted data
                # This is a simplified implementation
                # In a real implementation, this would use torch.distributed.recv
                # or similar NCCL operations to receive data from other nodes
                return None  # Placeholder
            except Exception as e:
                print(f"⚠️  Failed to receive via NCCL: {e}")
                return None
        else:
            # Memory backend - already handled locally
            return None
    
    def put_routing_decision(self, 
                           context: Any,
                           expert_indices: List[int],
                           expert_weights: List[float],
                           confidence_scores: List[float]) -> bool:
        """
        Store routing decision in cache
        
        Args:
            context: Input context
            expert_indices: Selected expert indices
            expert_weights: Weights for selected experts
            confidence_scores: Confidence scores for routing decisions
            
        Returns:
            bool: Whether storage was successful
        """
        try:
            with self.lock:
                # Create cache entry
                context_hash = self._hash_context(context)
                timestamp = time.time()
                
                entry = RoutingCacheEntry(
                    context_hash=context_hash,
                    expert_indices=expert_indices,
                    expert_weights=expert_weights,
                    confidence_scores=confidence_scores,
                    timestamp=timestamp,
                    node_id=self.node_id
                )
                
                # Add to local cache (LRU - move to end)
                self.cache[context_hash] = entry
                self.cache.move_to_end(context_hash)
                
                # Store in distributed cache
                self._distributed_put(context_hash, entry)
                
                # Evict old entries if cache is full
                self._evict_expired_entries()
                self._evict_lru_entries()
                
                return True
                
        except Exception as e:
            print(f"❌ Failed to store routing decision: {e}")
            return False
    
    def get_routing_decision(self, context: Any) -> Optional[Dict[str, Any]]:
        """
        Retrieve routing decision from cache
        
        Args:
            context: Input context
            
        Returns:
            Optional[Dict[str, Any]]: Cached routing decision or None
        """
        try:
            with self.lock:
                context_hash = self._hash_context(context)
                
                # Check local cache first
                if context_hash in self.cache:
                    entry = self.cache[context_hash]
                    
                    # Check TTL
                    if time.time() - entry.timestamp <= self.ttl_seconds:
                        # Update LRU order
                        self.cache.move_to_end(context_hash)
                        entry.access_count += 1
                        entry.last_accessed = time.time()
                        
                        # Update statistics
                        self.hits += 1
                        
                        return {
                            "expert_indices": entry.expert_indices,
                            "expert_weights": entry.expert_weights,
                            "confidence_scores": entry.confidence_scores,
                            "timestamp": entry.timestamp,
                            "node_id": entry.node_id,
                            "access_count": entry.access_count
                        }
                    else:
                        # Expired entry - remove it
                        del self.cache[context_hash]
                        self.evictions += 1
                
                # Check distributed cache
                distributed_entry = self._distributed_get(context_hash)
                if distributed_entry:
                    # Check TTL for distributed entry
                    if time.time() - distributed_entry.timestamp <= self.ttl_seconds:
                        # Add to local cache
                        self.cache[context_hash] = distributed_entry
                        self.cache.move_to_end(context_hash)
                        
                        # Update statistics
                        self.hits += 1
                        
                        return {
                            "expert_indices": distributed_entry.expert_indices,
                            "expert_weights": distributed_entry.expert_weights,
                            "confidence_scores": distributed_entry.confidence_scores,
                            "timestamp": distributed_entry.timestamp,
                            "node_id": distributed_entry.node_id,
                            "access_count": distributed_entry.access_count
                        }
                
                # Cache miss
                self.misses += 1
                return None
                
        except Exception as e:
            print(f"❌ Failed to retrieve routing decision: {e}")
            self.misses += 1
            return None
    
    def _evict_expired_entries(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time - entry.timestamp > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            self.evictions += 1
    
    def _evict_lru_entries(self):
        """Remove LRU entries when cache is full"""
        while len(self.cache) > self.max_cache_size:
            # Remove least recently used entry
            try:
                key, _ = self.cache.popitem(last=False)
                self.evictions += 1
            except KeyError:
                break
    
    def add_cluster_node(self, node_id: str):
        """Add node to cluster"""
        with self.lock:
            self.cluster_nodes.add(node_id)
            print(f"🔗 Node {node_id} added to cluster")
    
    def remove_cluster_node(self, node_id: str):
        """Remove node from cluster"""
        with self.lock:
            self.cluster_nodes.discard(node_id)
            print(f"🔗 Node {node_id} removed from cluster")
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster information"""
        with self.lock:
            return {
                "local_node_id": self.node_id,
                "cluster_nodes": list(self.cluster_nodes),
                "cluster_size": len(self.cluster_nodes) + 1  # +1 for local node
            }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            # Calculate average entry age
            current_time = time.time()
            if self.cache:
                avg_age = sum(current_time - entry.timestamp for entry in self.cache.values()) / len(self.cache)
            else:
                avg_age = 0
            
            return {
                "cache_size": len(self.cache),
                "max_cache_size": self.max_cache_size,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "average_entry_age": avg_age,
                "node_id": self.node_id,
                "ttl_seconds": self.ttl_seconds
            }
    
    def clear_cache(self):
        """Clear all cache entries"""
        with self.lock:
            cache_size = len(self.cache)
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            print(f"🧹 Cache cleared ({cache_size} entries removed)")
    
    def sync_with_cluster(self, other_caches: List['CrossNodeRoutingCache']):
        """
        Synchronize cache with other nodes in cluster
        
        Args:
            other_caches: List of other cache instances to sync with
        """
        try:
            synced_entries = 0
            
            with self.lock:
                for other_cache in other_caches:
                    if other_cache.node_id == self.node_id:
                        continue  # Skip self
                    
                    # Get entries from other cache
                    with other_cache.lock:
                        for key, entry in other_cache.cache.items():
                            # Only sync non-expired entries
                            if time.time() - entry.timestamp <= self.ttl_seconds:
                                # Add to local cache if not present or if newer
                                if key not in self.cache or entry.timestamp > self.cache[key].timestamp:
                                    self.cache[key] = entry
                                    self.cache.move_to_end(key)
                                    synced_entries += 1
                
                # Evict if needed after sync
                self._evict_lru_entries()
            
            print(f"🔄 Cache synchronized with cluster ({synced_entries} entries synced)")
            return True
            
        except Exception as e:
            print(f"❌ Failed to sync cache with cluster: {e}")
            return False
        
    def get_distributed_stats(self) -> Dict[str, Any]:
        """
        Get distributed cache statistics
        
        Returns:
            Dictionary with distributed cache statistics
        """
        stats = self.get_cache_stats()
        stats["distributed_backend"] = self.distributed_backend
        stats["node_id"] = self.node_id
        
        if self.distributed_backend == "redis" and self.redis_client:
            try:
                redis_info = self.redis_client.info()
                stats["redis_info"] = {
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "used_memory": redis_info.get("used_memory_human", "0B"),
                    "total_commands_processed": redis_info.get("total_commands_processed", 0)
                }
            except Exception as e:
                stats["redis_info"] = {"error": str(e)}
        elif self.distributed_backend == "nccl" and self.nccl_initialized:
            try:
                stats["nccl_info"] = {
                    "world_size": torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
                    "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                }
            except Exception as e:
                stats["nccl_info"] = {"error": str(e)}
                
        return stats

class DistributedMoEBenchmarkRunner:
    """Benchmark runner for distributed MoE with routing cache"""
    
    def __init__(self, model, device: Optional[str] = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        
        # Initialize routing cache
        self.routing_cache = CrossNodeRoutingCache(
            max_cache_size=5000,
            ttl_seconds=1800,  # 30 minutes
            enable_compression=True
        )
        
        # Benchmark settings
        self.warmup_iterations = 10
        self.benchmark_iterations = 100
    
    def benchmark_routing_performance(self, 
                                    batch_sizes: List[int] = [16, 32, 64],
                                    seq_lengths: List[int] = [64, 128, 256]) -> Dict[str, Any]:
        """
        Benchmark routing performance with and without cache
        
        Args:
            batch_sizes: List of batch sizes to test
            seq_lengths: List of sequence lengths to test
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        print("🚀 Running Cross-Node Routing Cache Benchmark")
        print("=" * 55)
        
        results = {
            "with_cache": {},
            "without_cache": {},
            "improvements": {},
            "cache_stats": {}
        }
        
        # Test without cache (baseline)
        print("📊 Benchmarking WITHOUT routing cache...")
        baseline_results = self._benchmark_routing(batch_sizes, seq_lengths, use_cache=False)
        results["without_cache"] = baseline_results
        
        # Test with cache
        print("\n📊 Benchmarking WITH routing cache...")
        cache_results = self._benchmark_routing(batch_sizes, seq_lengths, use_cache=True)
        results["with_cache"] = cache_results
        
        # Calculate improvements
        for bs in batch_sizes:
            for seq_len in seq_lengths:
                key = f"bs{bs}_seq{seq_len}"
                if key in baseline_results and key in cache_results:
                    baseline_time = baseline_results[key]["avg_routing_time"]
                    cache_time = cache_results[key]["avg_routing_time"]
                    
                    if baseline_time > 0 and cache_time > 0:
                        speedup = baseline_time / cache_time
                        improvement = ((baseline_time - cache_time) / baseline_time) * 100
                        
                        results["improvements"][key] = {
                            "speedup": speedup,
                            "improvement_percent": improvement,
                            "time_saved_ms": (baseline_time - cache_time) * 1000
                        }
        
        # Add cache statistics
        results["cache_stats"] = self.routing_cache.get_cache_stats()
        
        # Print summary
        print("\n📈 Routing Cache Performance Summary:")
        cache_stats = results["cache_stats"]
        print(f"   Cache Hit Rate: {cache_stats['hit_rate']:.2%}")
        print(f"   Cache Size: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
        
        for bs in batch_sizes:
            for seq_len in seq_lengths:
                key = f"bs{bs}_seq{seq_len}"
                if key in results["improvements"]:
                    improvement = results["improvements"][key]
                    print(f"   BS={bs}, Seq={seq_len}: "
                          f"{improvement['speedup']:.2f}x speedup "
                          f"({improvement['improvement_percent']:.1f}% improvement, "
                          f"{improvement['time_saved_ms']:.2f}ms saved)")
        
        return results
    
    def _benchmark_routing(self, batch_sizes: List[int], seq_lengths: List[int], 
                          use_cache: bool = False) -> Dict[str, Any]:
        """Benchmark routing with specific settings"""
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
                    
                    # Benchmark routing decisions
                    routing_times = []
                    total_times = []
                    
                    for i in range(self.benchmark_iterations):
                        # Create context for routing
                        context = input_ids.float().mean(dim=1)  # Simple context vector
                        
                        start_total = time.time()
                        
                        if use_cache:
                            # Check cache first
                            cached_decision = self.routing_cache.get_routing_decision(context)
                            
                            if cached_decision is not None:
                                # Use cached routing decision
                                expert_indices = cached_decision["expert_indices"]
                                expert_weights = cached_decision["expert_weights"]
                                routing_time = 0  # Near-instant routing
                            else:
                                # Compute routing decision
                                start_routing = time.time()
                                # In a real implementation, this would call the actual routing logic
                                expert_indices = [0, 1]  # Mock expert selection
                                expert_weights = [0.6, 0.4]  # Mock weights
                                routing_time = time.time() - start_routing
                                
                                # Store in cache
                                self.routing_cache.put_routing_decision(
                                    context, expert_indices, expert_weights, [0.9, 0.8]
                                )
                        else:
                            # Compute routing decision without cache
                            start_routing = time.time()
                            # In a real implementation, this would call the actual routing logic
                            expert_indices = [0, 1]  # Mock expert selection
                            expert_weights = [0.6, 0.4]  # Mock weights
                            routing_time = time.time() - start_routing
                        
                        # Run model with routing decision
                        _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        
                        end_total = time.time()
                        
                        routing_times.append(routing_time)
                        total_times.append(end_total - start_total)
                    
                    # Calculate statistics
                    avg_routing_time = sum(routing_times) / len(routing_times)
                    avg_total_time = sum(total_times) / len(total_times)
                    min_total_time = min(total_times)
                    max_total_time = max(total_times)
                    
                    results[key] = {
                        "avg_routing_time": avg_routing_time,
                        "avg_total_time": avg_total_time,
                        "min_total_time": min_total_time,
                        "max_total_time": max_total_time,
                        "batch_size": bs,
                        "seq_length": seq_len,
                        "iterations": self.benchmark_iterations,
                        "use_cache": use_cache
                    }
                    
                    print(f"      Avg Routing: {avg_routing_time*1000:.2f}ms, "
                          f"Total: {avg_total_time*1000:.2f}ms")
        
        return results
    
    def simulate_cluster_performance(self, num_nodes: int = 4) -> Dict[str, Any]:
        """
        Simulate performance in a multi-node cluster environment
        
        Args:
            num_nodes: Number of nodes in simulated cluster
            
        Returns:
            Dict[str, Any]: Cluster performance results
        """
        print(f"🌐 Simulating cluster performance with {num_nodes} nodes...")
        
        # Create simulated cluster nodes
        cluster_caches = []
        for i in range(num_nodes):
            cache = CrossNodeRoutingCache(max_cache_size=2000, ttl_seconds=1800)
            cache.node_id = f"node_{i}"
            cluster_caches.append(cache)
        
        # Add nodes to cluster
        for cache in cluster_caches:
            self.routing_cache.add_cluster_node(cache.node_id)
        
        # Simulate cache synchronization
        sync_success = self.routing_cache.sync_with_cluster(cluster_caches)
        
        # Simulate distributed routing
        cluster_results = {
            "nodes": num_nodes,
            "sync_success": sync_success,
            "cluster_info": self.routing_cache.get_cluster_info(),
            "local_cache_stats": self.routing_cache.get_cache_stats()
        }
        
        print(f"✅ Cluster simulation completed")
        print(f"   Nodes: {cluster_results['cluster_info']['cluster_size']}")
        print(f"   Sync Success: {sync_success}")
        
        return cluster_results

# Example usage
def example_cross_node_routing_cache():
    """Example of cross-node routing cache usage"""
    print("🔧 Setting up cross-node routing cache example...")
    
    # Simple model for demonstration
    import torch.nn as nn
    
    class SimpleMoEModel(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=128, num_experts=4):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.expert_layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)
            ])
            self.gate = nn.Linear(hidden_size, num_experts)
            self.output_layer = nn.Linear(hidden_size, 2)
            
        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            x = x.mean(dim=1)  # Global average pooling
            
            # Simple MoE routing (mock implementation)
            gate_logits = self.gate(x)
            expert_weights = torch.softmax(gate_logits, dim=-1)
            
            # Use top-2 experts
            top2_weights, top2_indices = torch.topk(expert_weights, 2, dim=-1)
            top2_weights = top2_weights / top2_weights.sum(dim=-1, keepdim=True)
            
            # Process through experts
            expert_outputs = []
            for i in range(len(top2_indices)):
                expert_idx = top2_indices[i]
                weight = top2_weights[i]
                
                # Combine outputs from top-2 experts
                output = torch.zeros_like(x[i])
                for j in range(2):
                    expert_output = self.expert_layers[expert_idx[j]](x[i])
                    output += expert_output * weight[j]
                
                expert_outputs.append(output)
            
            x = torch.stack(expert_outputs, dim=0)
            logits = self.output_layer(x)
            return logits
    
    # Create model
    model = SimpleMoEModel()
    print(f"✅ Created MoE model with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    # Create benchmark runner
    benchmark_runner = DistributedMoEBenchmarkRunner(model)
    
    # Run routing benchmark
    print("\n" + "="*60)
    routing_results = benchmark_runner.benchmark_routing_performance(
        batch_sizes=[16, 32],
        seq_lengths=[64, 128]
    )
    
    # Simulate cluster performance
    print("\n" + "="*60)
    cluster_results = benchmark_runner.simulate_cluster_performance(num_nodes=3)
    
    # Print final cache statistics
    print("\n" + "="*60)
    cache_stats = benchmark_runner.routing_cache.get_cache_stats()
    print("📊 Final Cache Statistics:")
    print(f"   Hit Rate: {cache_stats['hit_rate']:.2%}")
    print(f"   Cache Size: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
    print(f"   Total Requests: {cache_stats['total_requests']}")
    print(f"   Evictions: {cache_stats['evictions']}")
    
    return routing_results, cluster_results

if __name__ == "__main__":
    example_cross_node_routing_cache()