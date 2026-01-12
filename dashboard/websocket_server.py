"""
WebSocket Server for MAHIA Dashboard V3
Provides real-time streaming of training metrics
"""

import asyncio
import json
import websockets
import threading
import time
from typing import Dict, Any, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# FastAPI app
app = FastAPI(title="MAHIA Dashboard V3", version="3.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections
active_connections: List[WebSocket] = []

# In-memory storage for metrics
metrics_storage: Dict[str, List[Dict[str, Any]]] = {}

class MetricsBroadcaster:
    """Broadcasts metrics to all connected WebSocket clients"""
    
    def __init__(self):
        self.running = False
        self.broadcast_thread = None
        
    def start(self):
        """Start the broadcasting thread"""
        if not self.running:
            self.running = True
            self.broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True)
            self.broadcast_thread.start()
            print("📡 Metrics broadcaster started")
            
    def stop(self):
        """Stop the broadcasting thread"""
        self.running = False
        if self.broadcast_thread:
            self.broadcast_thread.join()
        print("🛑 Metrics broadcaster stopped")
        
    def _broadcast_loop(self):
        """Main broadcasting loop"""
        while self.running:
            try:
                # Broadcast current metrics to all connections
                if active_connections and metrics_storage:
                    # Get the latest metrics for each type
                    latest_metrics = {}
                    for metric_type, values in metrics_storage.items():
                        if values:
                            latest_metrics[metric_type] = values[-1]  # Get latest value
                    
                    if latest_metrics:
                        # Broadcast to all connections
                        message = json.dumps({
                            "type": "metrics_update",
                            "data": latest_metrics,
                            "timestamp": time.time()
                        })
                        
                        # Create a copy of active connections to avoid modification during iteration
                        connections_copy = active_connections.copy()
                        
                        for connection in connections_copy:
                            try:
                                asyncio.run_coroutine_threadsafe(
                                    connection.send_text(message), 
                                    asyncio.get_event_loop()
                                )
                            except Exception as e:
                                print(f"⚠️  Failed to send to connection: {e}")
                                
                time.sleep(0.1)  # Broadcast every 100ms
            except Exception as e:
                print(f"⚠️  Broadcast error: {e}")
                time.sleep(1)

# Initialize broadcaster
broadcaster = MetricsBroadcaster()

@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for metrics streaming"""
    await websocket.accept()
    active_connections.append(websocket)
    print(f"🔗 New WebSocket connection. Total: {len(active_connections)}")
    
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back received data
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print(f"🔗 WebSocket connection closed. Total: {len(active_connections)}")
    except Exception as e:
        print(f"⚠️  WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.post("/api/metrics")
async def post_metrics(metrics: Dict[str, Any]):
    """API endpoint to receive metrics from training process"""
    metric_type = metrics.get("type", "unknown")
    
    if metric_type not in metrics_storage:
        metrics_storage[metric_type] = []
        
    metrics_storage[metric_type].append(metrics)
    
    # Keep only last 1000 entries per metric type to prevent memory issues
    if len(metrics_storage[metric_type]) > 1000:
        metrics_storage[metric_type] = metrics_storage[metric_type][-1000:]
    
    return {"status": "success", "message": "Metrics received"}

@app.get("/api/metrics")
async def get_metrics():
    """API endpoint to get current metrics"""
    return metrics_storage

@app.get("/api/metrics/{metric_type}")
async def get_metric_type(metric_type: str):
    """API endpoint to get specific metric type"""
    return metrics_storage.get(metric_type, [])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MAHIA Dashboard V3 WebSocket Server", 
        "version": "3.0.0",
        "endpoints": {
            "WebSocket": "/ws/metrics",
            "Metrics POST": "/api/metrics",
            "Metrics GET": "/api/metrics",
            "Metrics by type": "/api/metrics/{metric_type}"
        }
    }

def start_server(host="localhost", port=8000):
    """Start the FastAPI server"""
    print(f"🚀 Starting MAHIA Dashboard V3 WebSocket Server on {host}:{port}")
    broadcaster.start()
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()