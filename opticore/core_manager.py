"""
CoreManager for MAHIA OptiCore
Central task scheduling and real-time control for OptiCore components.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Callable
from collections import deque
import queue

class CoreManager:
    """Central task scheduling and real-time control for OptiCore components"""
    
    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self.task_queue = queue.Queue(maxsize=max_queue_size)
        self.event_queue = queue.Queue(maxsize=max_queue_size)
        self.is_running = False
        self.worker_thread = None
        self.lock = threading.Lock()
        self.stats = {
            "tasks_processed": 0,
            "events_processed": 0,
            "queue_overflows": 0
        }
        
        print("🧠 CoreManager initialized")
        
    def start(self):
        """Start the core manager worker thread"""
        if self.is_running:
            print("⚠️  CoreManager is already running")
            return
            
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("⚡ CoreManager started")
        
    def stop(self):
        """Stop the core manager worker thread"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        print("⏹️  CoreManager stopped")
        
    def _worker_loop(self):
        """Main worker loop for processing tasks and events"""
        while self.is_running:
            try:
                # Process tasks
                try:
                    task = self.task_queue.get(timeout=0.1)
                    self._process_task(task)
                    self.task_queue.task_done()
                    with self.lock:
                        self.stats["tasks_processed"] += 1
                except queue.Empty:
                    pass
                
                # Process events
                try:
                    event = self.event_queue.get(timeout=0.1)
                    self._process_event(event)
                    self.event_queue.task_done()
                    with self.lock:
                        self.stats["events_processed"] += 1
                except queue.Empty:
                    pass
                    
            except Exception as e:
                print(f"❌ CoreManager worker error: {e}")
                
    def _process_task(self, task: Dict[str, Any]):
        """Process a task"""
        try:
            task_type = task.get("type", "unknown")
            handler = task.get("handler")
            params = task.get("params", {})
            
            if handler and callable(handler):
                handler(**params)
            else:
                print(f"⚠️  No handler for task type: {task_type}")
                
        except Exception as e:
            print(f"❌ Error processing task: {e}")
            
    def _process_event(self, event: Dict[str, Any]):
        """Process an event"""
        try:
            event_type = event.get("type", "unknown")
            handlers = event.get("handlers", [])
            
            for handler in handlers:
                if callable(handler):
                    handler(event)
                    
        except Exception as e:
            print(f"❌ Error processing event: {e}")
    
    def dispatch(self, task_type: str, handler: Callable, params: Optional[Dict[str, Any]] = None):
        """
        Dispatch a task to the core manager.
        
        Args:
            task_type: Type of task
            handler: Function to handle the task
            params: Parameters for the handler
        """
        task = {
            "type": task_type,
            "handler": handler,
            "params": params or {}
        }
        
        try:
            self.task_queue.put_nowait(task)
        except queue.Full:
            with self.lock:
                self.stats["queue_overflows"] += 1
            print("⚠️  Task queue overflow")
            
    def emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None, handlers: Optional[List[Callable]] = None):
        """
        Emit an event.
        
        Args:
            event_type: Type of event
            data: Event data
            handlers: List of handlers to process the event
        """
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data or {},
            "handlers": handlers or []
        }
        
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            with self.lock:
                self.stats["queue_overflows"] += 1
            print("⚠️  Event queue overflow")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get core manager statistics"""
        with self.lock:
            return self.stats.copy()
            
    def clear_stats(self):
        """Clear statistics"""
        with self.lock:
            self.stats = {
                "tasks_processed": 0,
                "events_processed": 0,
                "queue_overflows": 0
            }
        print("🗑️  CoreManager statistics cleared")

# Global instance
_core_manager = None

def get_core_manager() -> CoreManager:
    """Get the global core manager instance"""
    global _core_manager
    if _core_manager is None:
        _core_manager = CoreManager()
    return _core_manager

if __name__ == "__main__":
    # Example usage
    manager = get_core_manager()
    manager.start()
    
    # Define a sample task handler
    def sample_task(message: str):
        print(f"🔧 Executing task: {message}")
        time.sleep(0.1)  # Simulate work
        
    # Dispatch some tasks
    manager.dispatch("sample", sample_task, {"message": "Hello from OptiCore!"})
    manager.dispatch("sample", sample_task, {"message": "Another task"})
    
    # Wait a bit for processing
    time.sleep(1)
    
    # Print stats
    print(f"📊 Stats: {manager.get_stats()}")
    
    manager.stop()