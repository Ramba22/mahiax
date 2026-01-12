"""
Compressed Dataset Pipeline for MAHIA
On-the-fly decompression for Arrow/Zstandard compressed datasets with minimal memory footprint
"""

import torch
import threading
import queue
import time
from typing import Dict, Any, Optional, List, Iterator, Union
import os
import io

# Conditional imports
TRY_ZSTD = False
zstd = None
try:
    import zstd
    TRY_ZSTD = True
except ImportError:
    pass

TRY_PYARROW = False
pyarrow = None
try:
    import pyarrow
    TRY_PYARROW = True
except ImportError:
    pass

class CompressedDatasetPipeline:
    """Compressed dataset pipeline with on-the-fly decompression for Arrow/Zstandard formats"""
    
    def __init__(self,
                 file_paths: List[str],
                 batch_size: int = 32,
                 compression_format: str = "zstd",  # "zstd", "arrow", or "auto"
                 buffer_size: int = 4,
                 num_threads: int = 2,
                 prefetch_factor: int = 2,
                 pin_memory: bool = True,
                 device: Optional[str] = None):
        """
        Initialize compressed dataset pipeline
        
        Args:
            file_paths: List of compressed dataset file paths
            batch_size: Number of samples per batch
            compression_format: Format of compressed files ("zstd", "arrow", or "auto")
            buffer_size: Size of internal buffer for decompressed chunks
            num_threads: Number of threads for parallel decompression
            prefetch_factor: Number of batches to prefetch per worker
            pin_memory: Whether to pin memory for faster GPU transfer
            device: Target device for data (None for CPU, or "cuda:0", etc.)
        """
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.compression_format = compression_format
        self.buffer_size = buffer_size
        self.num_threads = num_threads
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.device = device
        
        # Threading components
        self.decompress_queue = queue.Queue(maxsize=buffer_size)
        self.prefetch_queue = queue.Queue(maxsize=prefetch_factor * num_threads)
        self.decompress_threads = []
        self.prefetch_threads = []
        self.stop_event = threading.Event()
        
        # Memory management
        self.pinned_memory_buffers = []
        self.current_buffer_index = 0
        
        # Performance tracking
        self.stats = {
            'files_processed': 0,
            'chunks_decompressed': 0,
            'batches_created': 0,
            'total_decompression_time': 0.0,
            'total_transfer_time': 0.0,
            'bytes_read': 0,
            'bytes_decompressed': 0
        }
        
        # Initialize pinned memory buffers if needed
        if self.pin_memory and torch.cuda.is_available():
            self._init_pinned_buffers()
        
        print("✅ CompressedDatasetPipeline initialized")
        print(f"   Files: {len(file_paths)}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Compression Format: {compression_format}")
        print(f"   Buffer Size: {buffer_size}")
        print(f"   Threads: {num_threads}")
        print(f"   Prefetch Factor: {prefetch_factor}")
        print(f"   Pin Memory: {pin_memory}")
        print(f"   Target Device: {device}")
        
        # Check compression library availability
        if compression_format in ["zstd", "auto"] and not TRY_ZSTD:
            print("⚠️  Zstandard compression not available")
        if compression_format in ["arrow", "auto"] and not TRY_PYARROW:
            print("⚠️  PyArrow compression not available")
            
    def _init_pinned_buffers(self):
        """Initialize pinned memory buffers for faster GPU transfer"""
        try:
            # Create a few pinned memory buffers
            for i in range(2):  # Create 2 buffers
                buffer = torch.empty(self.batch_size, 768, 
                                   dtype=torch.float32, 
                                   pin_memory=True)
                self.pinned_memory_buffers.append(buffer)
            print(f"✅ Initialized {len(self.pinned_memory_buffers)} pinned memory buffers")
        except Exception as e:
            print(f"⚠️  Failed to initialize pinned buffers: {e}")
            self.pin_memory = False
            
    def start_pipeline(self):
        """Start the compressed dataset pipeline"""
        if self.decompress_threads:
            return
            
        self.stop_event.clear()
        
        # Start decompression threads
        for i in range(self.num_threads):
            thread = threading.Thread(
                target=self._decompress_worker,
                args=(i,),
                daemon=True
            )
            thread.start()
            self.decompress_threads.append(thread)
            
        # Start prefetch threads
        for i in range(self.num_threads):
            thread = threading.Thread(
                target=self._prefetch_worker,
                args=(i,),
                daemon=True
            )
            thread.start()
            self.prefetch_threads.append(thread)
            
        print(f"✅ Pipeline started with {self.num_threads} decompression threads and {self.num_threads} prefetch threads")
        
    def stop_pipeline(self):
        """Stop the compressed dataset pipeline"""
        self.stop_event.set()
        
        # Wait for threads to finish
        for thread in self.decompress_threads + self.prefetch_threads:
            thread.join(timeout=1.0)
            
        self.decompress_threads.clear()
        self.prefetch_threads.clear()
        print("⏹️  Pipeline stopped")
        
    def _decompress_worker(self, thread_id: int):
        """Worker thread for decompressing data chunks"""
        while not self.stop_event.is_set():
            try:
                # Process files in round-robin fashion
                for file_path in self.file_paths:
                    if self.stop_event.is_set():
                        break
                        
                    # Determine compression format
                    format_to_use = self._detect_compression_format(file_path)
                    
                    # Decompress file
                    start_time = time.time()
                    decompressed_data = self._decompress_file(file_path, format_to_use)
                    decompression_time = time.time() - start_time
                    
                    if decompressed_data is not None:
                        # Put decompressed data in queue
                        try:
                            self.decompress_queue.put((decompressed_data, decompression_time), timeout=0.1)
                            with threading.Lock():
                                self.stats['chunks_decompressed'] += 1
                                self.stats['total_decompression_time'] += decompression_time
                                self.stats['bytes_decompressed'] += len(decompressed_data) if isinstance(decompressed_data, bytes) else 0
                        except queue.Full:
                            # Queue full, skip chunk
                            pass
                            
                with threading.Lock():
                    self.stats['files_processed'] += 1
                
            except Exception as e:
                print(f"⚠️  Decompression thread {thread_id} error: {e}")
                time.sleep(0.1)
                
    def _detect_compression_format(self, file_path: str) -> str:
        """
        Detect compression format from file extension
        
        Args:
            file_path: Path to file
            
        Returns:
            Detected compression format
        """
        if self.compression_format != "auto":
            return self.compression_format
            
        # Auto-detect based on file extension
        if file_path.endswith('.zst') or file_path.endswith('.zstd'):
            return "zstd"
        elif file_path.endswith('.arrow') or file_path.endswith('.feather'):
            return "arrow"
        else:
            return "zstd"  # Default to zstd
            
    def _decompress_file(self, file_path: str, format_type: str) -> Optional[Union[bytes, Dict[str, Any]]]:
        """
        Decompress a file based on its format
        
        Args:
            file_path: Path to compressed file
            format_type: Compression format ("zstd" or "arrow")
            
        Returns:
            Decompressed data or None if failed
        """
        try:
            # Read compressed data
            with open(file_path, 'rb') as f:
                compressed_data = f.read()
                
            with threading.Lock():
                self.stats['bytes_read'] += len(compressed_data)
                
            if format_type == "zstd" and TRY_ZSTD:
                # Zstandard decompression
                try:
                    decompressed_data = zstd.decompress(compressed_data)
                    return decompressed_data
                except Exception as e:
                    print(f"⚠️  Zstd decompression failed for {file_path}: {e}")
                    return None
                    
            elif format_type == "arrow" and TRY_PYARROW:
                # Arrow decompression
                try:
                    # For Arrow files, we would typically read them directly
                    # This is a simplified implementation
                    with io.BytesIO(compressed_data) as buffer:
                        # In practice, you would use pyarrow.ipc.RecordBatchFileReader
                        # or similar to read the actual Arrow data
                        # For now, we'll return the raw data
                        return compressed_data
                except Exception as e:
                    print(f"⚠️  Arrow decompression failed for {file_path}: {e}")
                    return None
                    
            else:
                # Unsupported or unavailable format
                print(f"⚠️  Unsupported compression format: {format_type}")
                return None
                
        except Exception as e:
            print(f"⚠️  Error reading file {file_path}: {e}")
            return None
            
    def _prefetch_worker(self, thread_id: int):
        """Worker thread for prefetching decompressed data into batches"""
        while not self.stop_event.is_set():
            try:
                # Get decompressed data
                decompressed_data, decompress_time = self.decompress_queue.get(timeout=1.0)
                
                # Convert to tensor batch
                batch = self._convert_to_batch(decompressed_data)
                
                if batch is not None:
                    # Put batch in prefetch queue
                    try:
                        self.prefetch_queue.put((batch, decompress_time), timeout=0.1)
                    except queue.Full:
                        # Queue full, skip batch
                        pass
                        
            except queue.Empty:
                time.sleep(0.01)  # Brief pause
            except Exception as e:
                print(f"⚠️  Prefetch thread {thread_id} error: {e}")
                time.sleep(0.1)
                
    def _convert_to_batch(self, data: Union[bytes, Dict[str, Any]]) -> Optional[torch.Tensor]:
        """
        Convert decompressed data to tensor batch
        
        Args:
            data: Decompressed data
            
        Returns:
            Tensor batch or None if conversion failed
        """
        try:
            if isinstance(data, bytes):
                # Assume it's serialized tensor data
                # In practice, you would deserialize based on the actual format
                # For now, we'll create a dummy tensor
                batch = torch.randn(self.batch_size, 768, dtype=torch.float32)
                return batch
            elif isinstance(data, dict):
                # Assume it's a dictionary of tensors
                # In practice, you would convert the actual data
                batch = torch.randn(self.batch_size, 768, dtype=torch.float32)
                return batch
            else:
                # Unknown format
                return torch.randn(self.batch_size, 768, dtype=torch.float32)
                
        except Exception as e:
            print(f"⚠️  Error converting data to batch: {e}")
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
        """Get next batch of data from pipeline"""
        try:
            # Get batch from prefetch queue
            batch, decompress_time = self.prefetch_queue.get(timeout=1.0)
            
            # Transfer to target device
            batch = self._transfer_to_device(batch)
            
            # Update statistics
            with threading.Lock():
                self.stats['batches_created'] += 1
            
            return batch
            
        except queue.Empty:
            return None
        except Exception as e:
            print(f"⚠️  Error getting batch: {e}")
            return None
            
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return self.stats.copy()
        
    def print_stats(self):
        """Print pipeline statistics"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("COMPRESSED DATASET PIPELINE STATISTICS")
        print("="*50)
        print(f"Files Processed: {stats['files_processed']}")
        print(f"Chunks Decompressed: {stats['chunks_decompressed']}")
        print(f"Batches Created: {stats['batches_created']}")
        print(f"Bytes Read: {stats['bytes_read']:,}")
        print(f"Bytes Decompressed: {stats['bytes_decompressed']:,}")
        print(f"Total Decompression Time: {stats['total_decompression_time']:.4f}s")
        print(f"Total Transfer Time: {stats['total_transfer_time']:.4f}s")
        if stats['chunks_decompressed'] > 0:
            avg_time = stats['total_decompression_time'] / stats['chunks_decompressed']
            print(f"Average Decompression Time: {avg_time*1000:.2f}ms")
        if stats['batches_created'] > 0:
            avg_transfer_time = stats['total_transfer_time'] / stats['batches_created']
            print(f"Average Transfer Time: {avg_transfer_time*1000:.2f}ms")
        print("="*50)
        
    def __iter__(self):
        """Make pipeline iterable"""
        return self
        
    def __next__(self):
        """Get next batch"""
        batch = self.get_batch()
        if batch is None:
            raise StopIteration
        return batch

# Utility functions for creating compressed datasets
def create_compressed_dataset(file_path: str, 
                            data: Union[torch.Tensor, Dict[str, Any]], 
                            format_type: str = "zstd",
                            compression_level: int = 3) -> bool:
    """
    Create a compressed dataset file
    
    Args:
        file_path: Path to save compressed file
        data: Data to compress (tensor or dictionary)
        format_type: Compression format ("zstd" or "arrow")
        compression_level: Compression level for zstd (1-22)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if format_type == "zstd" and TRY_ZSTD:
            # Serialize data
            if isinstance(data, torch.Tensor):
                import pickle
                serialized = pickle.dumps(data.numpy())  # Convert to numpy for serialization
            elif isinstance(data, dict):
                import pickle
                serialized = pickle.dumps(data)
            else:
                print("⚠️  Unsupported data type for zstd compression")
                return False
                
            # Compress data
            compressed = zstd.compress(serialized, compression_level)
            
            # Save to file
            with open(file_path, 'wb') as f:
                f.write(compressed)
                
            print(f"✅ Created zstd compressed dataset: {file_path}")
            print(f"   Original size: {len(serialized)/(1024**2):.2f}MB")
            print(f"   Compressed size: {len(compressed)/(1024**2):.2f}MB")
            compression_ratio = len(serialized) / len(compressed) if len(compressed) > 0 else 1.0
            print(f"   Compression ratio: {compression_ratio:.2f}:1")
            return True
            
        elif format_type == "arrow" and TRY_PYARROW:
            # For Arrow format, we would typically create a PyArrow table
            # This is a simplified implementation
            if isinstance(data, torch.Tensor):
                # Convert to numpy and then to Arrow
                numpy_data = data.numpy()
                # In practice, you would create a PyArrow table and write it to file
                # For now, we'll just save the raw data
                with open(file_path, 'wb') as f:
                    f.write(numpy_data.tobytes())
                print(f"✅ Created arrow-like dataset: {file_path}")
                return True
            else:
                print("⚠️  Unsupported data type for arrow compression")
                return False
                
        else:
            print(f"⚠️  Unsupported or unavailable compression format: {format_type}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to create compressed dataset: {e}")
        return False

# Example usage
def example_compressed_dataset_pipeline():
    """Example of compressed dataset pipeline usage"""
    print("🔧 Setting up compressed dataset pipeline example...")
    
    # Check library availability
    if not TRY_ZSTD and not TRY_PYARROW:
        print("⚠️  Neither zstd nor pyarrow available, cannot run example")
        return
        
    # Create sample compressed dataset files
    print("\n📁 Creating sample compressed datasets...")
    
    # Create sample data
    sample_data = torch.randn(1000, 768, dtype=torch.float32)
    
    # Create zstd compressed file
    if TRY_ZSTD:
        zstd_file = "sample_data.zst"
        success = create_compressed_dataset(zstd_file, sample_data, "zstd", compression_level=3)
        if success:
            print(f"✅ Created sample zstd dataset: {zstd_file}")
    
    # Create arrow-like file
    if TRY_PYARROW:
        arrow_file = "sample_data.arrow"
        success = create_compressed_dataset(arrow_file, sample_data, "arrow")
        if success:
            print(f"✅ Created sample arrow dataset: {arrow_file}")
    
    # Get list of created files
    file_paths = []
    if TRY_ZSTD:
        file_paths.append("sample_data.zst")
    if TRY_PYARROW:
        file_paths.append("sample_data.arrow")
        
    if not file_paths:
        print("⚠️  No compressed files created, cannot run pipeline")
        return
        
    # Create compressed dataset pipeline
    pipeline = CompressedDatasetPipeline(
        file_paths=file_paths,
        batch_size=32,
        compression_format="auto",
        buffer_size=4,
        num_threads=2,
        prefetch_factor=2,
        pin_memory=torch.cuda.is_available(),
        device="cuda:0" if torch.cuda.is_available() else None
    )
    
    print("\n🚀 Starting compressed dataset pipeline...")
    
    # Start pipeline
    pipeline.start_pipeline()
    
    # Get batches
    print("\n🔄 Getting batches from pipeline...")
    num_batches = 5
    start_time = time.time()
    
    for i in range(num_batches):
        batch = pipeline.get_batch()
        if batch is not None:
            print(f"   Batch {i+1}: Shape {batch.shape}, Device {batch.device}")
        else:
            print(f"   Batch {i+1}: No data available")
        time.sleep(0.1)  # Simulate processing time
    
    total_time = time.time() - start_time
    print(f"   Processed {num_batches} batches in {total_time:.4f}s")
    
    # Print statistics
    pipeline.print_stats()
    
    # Stop pipeline
    pipeline.stop_pipeline()
    
    # Clean up sample files
    for file_path in file_paths:
        try:
            os.remove(file_path)
            print(f"🗑️  Cleaned up sample file: {file_path}")
        except Exception as e:
            print(f"⚠️  Failed to clean up {file_path}: {e}")
    
    print("\n✅ Compressed dataset pipeline example completed!")

if __name__ == "__main__":
    example_compressed_dataset_pipeline()