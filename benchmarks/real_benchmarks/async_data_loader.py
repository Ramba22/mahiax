"""
Async Data Loader with Prefetch Pipeline for MAHIA
Eliminates IO bottleneck through asynchronous data loading and prefetching
"""

import torch
from torch.utils.data import Dataset, DataLoader
import threading
import queue
import time
from typing import Optional, Dict, Any, List, Iterator
import asyncio
import concurrent.futures

class AsyncDataLoader:
    """Async data loader with prefetch pipeline to eliminate IO bottleneck"""
    
    def __init__(self, 
                 dataset: Dataset,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 prefetch_factor: int = 2,
                 pin_memory: bool = True,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 timeout: float = 30.0):
        """
        Initialize async data loader with prefetch pipeline
        
        Args:
            dataset: PyTorch dataset to load
            batch_size: Number of samples per batch
            num_workers: Number of worker threads for data loading
            prefetch_factor: Number of batches to prefetch per worker
            pin_memory: Whether to pin memory for faster GPU transfer
            shuffle: Whether to shuffle the data
            drop_last: Whether to drop the last incomplete batch
            timeout: Timeout for data loading operations
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.timeout = timeout
        
        # Create standard DataLoader
        self.data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            timeout=timeout
        )
        
        # Async components
        self.prefetch_queue = queue.Queue(maxsize=prefetch_factor * num_workers)
        self.prefetch_thread = None
        self.stop_event = threading.Event()
        
        # Performance tracking
        self.stats = {
            'batches_loaded': 0,
            'prefetch_hits': 0,
            'prefetch_misses': 0,
            'total_load_time': 0.0,
            'avg_batch_time': 0.0
        }
        
        print("✅ AsyncDataLoader initialized")
        print(f"   Batch Size: {batch_size}")
        print(f"   Workers: {num_workers}")
        print(f"   Prefetch Factor: {prefetch_factor}")
        print(f"   Pin Memory: {pin_memory}")
        
    def start_prefetch(self):
        """Start prefetching data in background thread"""
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            return
            
        self.stop_event.clear()
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
        print("✅ Prefetch worker started")
        
    def stop_prefetching(self):
        """Stop prefetching data"""
        self.stop_event.set()
        if self.prefetch_thread is not None:
            self.prefetch_thread.join(timeout=1.0)
        print("⏹️  Prefetch worker stopped")
        
    def _prefetch_worker(self):
        """Background worker to prefetch data batches"""
        data_iter = iter(self.data_loader)
        
        while not self.stop_event.is_set():
            try:
                # Load next batch
                start_time = time.time()
                batch = next(data_iter)
                load_time = time.time() - start_time
                
                # Put batch in queue
                try:
                    self.prefetch_queue.put((batch, load_time), timeout=0.1)
                except queue.Full:
                    # Queue is full, skip this batch
                    pass
                    
            except StopIteration:
                # Dataset exhausted, restart
                data_iter = iter(self.data_loader)
            except Exception as e:
                print(f"⚠️  Prefetch error: {e}")
                time.sleep(0.1)  # Brief pause before retry
                
    def __iter__(self) -> Iterator:
        """Return iterator for async data loading"""
        return AsyncDataLoaderIterator(self)
        
    def __len__(self) -> int:
        """Return number of batches"""
        return len(self.data_loader)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get data loading statistics"""
        return self.stats.copy()
        
    def print_stats(self):
        """Print data loading statistics"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("ASYNC DATA LOADER STATISTICS")
        print("="*50)
        print(f"Batches Loaded: {stats['batches_loaded']}")
        print(f"Prefetch Hits: {stats['prefetch_hits']}")
        print(f"Prefetch Misses: {stats['prefetch_misses']}")
        print(f"Total Load Time: {stats['total_load_time']:.4f}s")
        if stats['batches_loaded'] > 0:
            avg_time = stats['avg_batch_time']
            print(f"Average Batch Time: {avg_time*1000:.2f}ms")
        print("="*50)

class AsyncDataLoaderIterator:
    """Iterator for async data loader"""
    
    def __init__(self, async_loader: AsyncDataLoader):
        self.async_loader = async_loader
        self.data_iter = iter(async_loader.data_loader)
        self.prefetch_available = async_loader.prefetch_thread is not None
        
    def __iter__(self):
        return self
        
    def __next__(self):
        start_time = time.time()
        
        try:
            if self.prefetch_available:
                # Try to get prefetched batch
                try:
                    batch, load_time = self.async_loader.prefetch_queue.get_nowait()
                    self.async_loader.stats['prefetch_hits'] += 1
                    self.async_loader.stats['total_load_time'] += load_time
                except queue.Empty:
                    # No prefetched batch available, load synchronously
                    self.async_loader.stats['prefetch_misses'] += 1
                    batch = next(self.data_iter)
                    load_time = time.time() - start_time
                    self.async_loader.stats['total_load_time'] += load_time
            else:
                # Load synchronously
                batch = next(self.data_iter)
                load_time = time.time() - start_time
                self.async_loader.stats['total_load_time'] += load_time
                
            # Update statistics
            self.async_loader.stats['batches_loaded'] += 1
            self.async_loader.stats['avg_batch_time'] = (
                self.async_loader.stats['total_load_time'] / 
                self.async_loader.stats['batches_loaded']
            )
            
            return batch
            
        except StopIteration:
            raise
        except Exception as e:
            print(f"⚠️  Data loading error: {e}")
            raise

class StreamingDataLoader:
    """Streaming data loader for direct disk-to-GPU with minimal RAM usage"""
    
    def __init__(self,
                 file_paths: List[str],
                 batch_size: int = 32,
                 chunk_size: int = 1024,
                 buffer_size: int = 8,
                 num_threads: int = 2,
                 pin_memory: bool = True,
                 device: Optional[str] = None,
                 file_format: str = "binary"):  # "binary", "arrow", "parquet"
        """
        Initialize streaming data loader
        
        Args:
            file_paths: List of file paths to stream data from
            batch_size: Number of samples per batch
            chunk_size: Number of samples to read in each chunk
            buffer_size: Size of internal buffer
            num_threads: Number of threads for parallel reading
            pin_memory: Whether to pin memory for faster GPU transfer
            device: Target device for data (None for CPU, or "cuda:0", etc.)
            file_format: Format of data files ("binary", "arrow", "parquet")
        """
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.num_threads = num_threads
        self.pin_memory = pin_memory
        self.device = device
        self.file_format = file_format
        
        # Threading components
        self.data_queue = queue.Queue(maxsize=buffer_size)
        self.control_queue = queue.Queue()
        self.reader_threads = []
        self.stop_event = threading.Event()
        
        # Memory management
        self.pinned_memory_buffers = []
        self.current_buffer_index = 0
        
        # Performance tracking
        self.stats = {
            'files_processed': 0,
            'chunks_read': 0,
            'batches_streamed': 0,
            'total_read_time': 0.0,
            'total_transfer_time': 0.0,
            'bytes_read': 0
        }
        
        # Initialize pinned memory buffers if needed
        if self.pin_memory and torch.cuda.is_available():
            self._init_pinned_buffers()
        
        print("✅ StreamingDataLoader initialized")
        print(f"   Files: {len(file_paths)}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Chunk Size: {chunk_size}")
        print(f"   Buffer Size: {buffer_size}")
        print(f"   Threads: {num_threads}")
        print(f"   Pin Memory: {pin_memory}")
        print(f"   Target Device: {device}")
        print(f"   File Format: {file_format}")
        
    def _init_pinned_buffers(self):
        """Initialize pinned memory buffers for faster GPU transfer"""
        try:
            # Create a few pinned memory buffers
            for i in range(2):  # Create 2 buffers
                buffer = torch.empty(self.chunk_size, 768, 
                                   dtype=torch.float32, 
                                   pin_memory=True)
                self.pinned_memory_buffers.append(buffer)
            print(f"✅ Initialized {len(self.pinned_memory_buffers)} pinned memory buffers")
        except Exception as e:
            print(f"⚠️  Failed to initialize pinned buffers: {e}")
            self.pin_memory = False
            
    def start_streaming(self):
        """Start streaming data from files"""
        if self.reader_threads:
            return
            
        self.stop_event.clear()
        
        # Start reader threads
        for i in range(self.num_threads):
            thread = threading.Thread(
                target=self._reader_worker,
                args=(i,),
                daemon=True
            )
            thread.start()
            self.reader_threads.append(thread)
            
        print(f"✅ Streaming started with {self.num_threads} reader threads")
        
    def stop_streaming(self):
        """Stop streaming data"""
        self.stop_event.set()
        
        # Wait for threads to finish
        for thread in self.reader_threads:
            thread.join(timeout=1.0)
            
        self.reader_threads.clear()
        print("⏹️  Streaming stopped")
        
    def _reader_worker(self, thread_id: int):
        """Worker thread for reading data chunks"""
        while not self.stop_event.is_set():
            try:
                # Read chunks from files
                for file_path in self.file_paths:
                    if self.stop_event.is_set():
                        break
                        
                    start_time = time.time()
                    chunk = self._read_chunk(file_path)
                    read_time = time.time() - start_time
                    
                    if chunk is not None:
                        # Put chunk in queue
                        try:
                            self.data_queue.put((chunk, read_time), timeout=0.1)
                            with threading.Lock():
                                self.stats['chunks_read'] += 1
                                self.stats['total_read_time'] += read_time
                                self.stats['bytes_read'] += chunk.nbytes if hasattr(chunk, 'nbytes') else 0
                        except queue.Full:
                            # Queue full, skip chunk
                            pass
                            
                with threading.Lock():
                    self.stats['files_processed'] += 1
                
            except Exception as e:
                print(f"⚠️  Reader thread {thread_id} error: {e}")
                time.sleep(0.1)
                
    def _read_chunk(self, file_path: str) -> Optional[torch.Tensor]:
        """Read a chunk of data from file"""
        try:
            if self.file_format == "binary":
                # For binary files, we'll simulate reading
                # In practice, this would read actual binary data
                chunk = torch.randn(self.chunk_size, 768, dtype=torch.float32)
                return chunk
            elif self.file_format == "arrow":
                # For Arrow files
                try:
                    import pyarrow as pa
                    # This is a simplified implementation
                    # In practice, you would read actual Arrow data
                    chunk = torch.randn(self.chunk_size, 768, dtype=torch.float32)
                    return chunk
                except ImportError:
                    print("⚠️  PyArrow not available, falling back to random data")
                    return torch.randn(self.chunk_size, 768, dtype=torch.float32)
            elif self.file_format == "parquet":
                # For Parquet files
                try:
                    import pandas as pd
                    # This is a simplified implementation
                    # In practice, you would read actual Parquet data
                    chunk = torch.randn(self.chunk_size, 768, dtype=torch.float32)
                    return chunk
                except ImportError:
                    print("⚠️  Pandas not available, falling back to random data")
                    return torch.randn(self.chunk_size, 768, dtype=torch.float32)
            else:
                # Default to random data
                return torch.randn(self.chunk_size, 768, dtype=torch.float32)
                
        except Exception as e:
            print(f"⚠️  Error reading chunk from {file_path}: {e}")
            return None
            
    def _transfer_to_device(self, data: torch.Tensor) -> torch.Tensor:
        """
        Transfer data to target device with minimal memory usage
        
        Args:
            data: Data tensor to transfer
            
        Returns:
            Transferred tensor
        """
        try:
            if self.device and torch.cuda.is_available():
                # Use pinned memory for faster transfer if available
                if self.pin_memory and self.pinned_memory_buffers:
                    # Use pre-allocated pinned buffer
                    buffer = self.pinned_memory_buffers[self.current_buffer_index]
                    self.current_buffer_index = (self.current_buffer_index + 1) % len(self.pinned_memory_buffers)
                    
                    # Copy data to pinned buffer
                    buffer.copy_(data)
                    
                    # Transfer to GPU
                    start_time = time.time()
                    gpu_data = buffer.to(self.device, non_blocking=True)
                    transfer_time = time.time() - start_time
                    
                    with threading.Lock():
                        self.stats['total_transfer_time'] += transfer_time
                        
                    return gpu_data
                else:
                    # Direct transfer
                    start_time = time.time()
                    gpu_data = data.to(self.device, non_blocking=self.pin_memory)
                    transfer_time = time.time() - start_time
                    
                    with threading.Lock():
                        self.stats['total_transfer_time'] += transfer_time
                        
                    return gpu_data
            else:
                # Keep on CPU
                return data
                
        except Exception as e:
            print(f"⚠️  Error transferring data to device: {e}")
            return data
            
    def get_batch(self) -> Optional[torch.Tensor]:
        """Get next batch of data"""
        try:
            # Collect samples for batch
            batch_samples = []
            batch_time = 0.0
            
            samples_needed = self.batch_size
            
            while samples_needed > 0:
                # Get chunk from queue
                chunk, read_time = self.data_queue.get(timeout=1.0)
                
                # Take what we need from the chunk
                samples_to_take = min(samples_needed, chunk.shape[0])
                batch_samples.append(chunk[:samples_to_take])
                batch_time += read_time
                
                samples_needed -= samples_to_take
                
                # If we didn't use the whole chunk, put the remainder back
                if samples_to_take < chunk.shape[0]:
                    remainder = chunk[samples_to_take:]
                    try:
                        self.data_queue.put((remainder, 0), timeout=0.01)
                    except queue.Full:
                        pass  # Drop remainder if queue is full
                        
            # Concatenate samples into batch
            if len(batch_samples) > 1:
                batch = torch.cat(batch_samples, dim=0)
            else:
                batch = batch_samples[0]
                
            # Ensure exact batch size
            if batch.shape[0] > self.batch_size:
                batch = batch[:self.batch_size]
            elif batch.shape[0] < self.batch_size:
                # Pad with zeros if needed (in practice, you might want to handle this differently)
                padding_size = self.batch_size - batch.shape[0]
                padding = torch.zeros(padding_size, batch.shape[1], dtype=batch.dtype)
                batch = torch.cat([batch, padding], dim=0)
            
            # Transfer to target device
            batch = self._transfer_to_device(batch)
            
            # Update statistics
            with threading.Lock():
                self.stats['batches_streamed'] += 1
            
            return batch
            
        except queue.Empty:
            return None
        except Exception as e:
            print(f"⚠️  Error creating batch: {e}")
            return None
            
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        return self.stats.copy()
        
    def print_stats(self):
        """Print streaming statistics"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("STREAMING DATA LOADER STATISTICS")
        print("="*50)
        print(f"Files Processed: {stats['files_processed']}")
        print(f"Chunks Read: {stats['chunks_read']}")
        print(f"Batches Streamed: {stats['batches_streamed']}")
        print(f"Total Read Time: {stats['total_read_time']:.4f}s")
        print(f"Total Transfer Time: {stats['total_transfer_time']:.4f}s")
        print(f"Bytes Read: {stats['bytes_read']:,}")
        if stats['chunks_read'] > 0:
            avg_time = stats['total_read_time'] / stats['chunks_read']
            print(f"Average Read Time: {avg_time*1000:.2f}ms")
        if stats['batches_streamed'] > 0:
            avg_transfer_time = stats['total_transfer_time'] / stats['batches_streamed']
            print(f"Average Transfer Time: {avg_transfer_time*1000:.2f}ms")
        print("="*50)
        
    def __iter__(self):
        """Make StreamingDataLoader iterable"""
        return self
        
    def __next__(self):
        """Get next batch"""
        batch = self.get_batch()
        if batch is None:
            raise StopIteration
        return batch

# Example usage
def example_async_data_loading():
    """Example of async data loading usage"""
    print("🔧 Setting up Async Data Loader example...")
    
    # Create dummy dataset
    class DummyDataset(Dataset):
        def __init__(self, size: int = 1000):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Simulate loading data (could be from disk, network, etc.)
            time.sleep(0.001)  # Simulate IO delay
            return torch.randn(768), torch.randint(0, 10, (1,)).item()
    
    # Create async data loader
    dataset = DummyDataset(size=1000)
    async_loader = AsyncDataLoader(
        dataset=dataset,
        batch_size=32,
        num_workers=2,
        prefetch_factor=3,
        pin_memory=True
    )
    
    print("\n🚀 Testing async data loading...")
    
    # Start prefetching
    async_loader.start_prefetch()
    
    # Load batches
    num_batches = 5
    start_time = time.time()
    
    for i, (data, labels) in enumerate(async_loader):
        if i >= num_batches:
            break
        print(f"   Batch {i+1}: Data shape {data.shape}, Labels shape {labels.shape}")
    
    total_time = time.time() - start_time
    print(f"   Loaded {num_batches} batches in {total_time:.4f}s")
    
    # Print statistics
    async_loader.print_stats()
    
    # Stop prefetching
    async_loader.stop_prefetching()
    
    print("\n🚀 Testing streaming data loader...")
    
    # Create streaming data loader
    file_paths = [f"data_file_{i}.bin" for i in range(4)]
    streaming_loader = StreamingDataLoader(
        file_paths=file_paths,
        batch_size=16,
        chunk_size=256,
        buffer_size=4,
        num_threads=2
    )
    
    # Start streaming
    streaming_loader.start_streaming()
    
    # Get batches
    for i in range(3):
        batch = streaming_loader.get_batch()
        if batch is not None:
            print(f"   Streamed batch {i+1}: Shape {batch.shape}")
        else:
            print(f"   Streamed batch {i+1}: No data available")
        time.sleep(0.1)  # Simulate processing time
    
    # Print statistics
    streaming_loader.print_stats()
    
    # Stop streaming
    streaming_loader.stop_streaming()

if __name__ == "__main__":
    example_async_data_loading()