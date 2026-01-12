# MAHIA Dashboard V3

Real-time monitoring dashboard for MAHIA training with WebSocket streaming.

## Features

- Real-time metrics streaming via WebSocket
- Interactive visualizations with Streamlit
- GPU utilization monitoring
- Loss curve tracking
- Learning rate visualization
- Entropy and confidence metrics
- Controller action monitoring

## Components

1. **WebSocket Server** (`websocket_server.py`): FastAPI backend that streams metrics to clients
2. **Streamlit Frontend** (`dashboard_frontend.py`): Interactive dashboard interface
3. **Telemetry Integration** (`telemetry_integration.py`): Bridge between MAHIA telemetry system and dashboard

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Start the WebSocket server:
   ```bash
   python websocket_server.py
   ```

2. In another terminal, start the Streamlit dashboard:
   ```bash
   streamlit run dashboard_frontend.py
   ```

3. The dashboard will be available at `http://localhost:8501`

## API Endpoints

- WebSocket: `ws://localhost:8000/ws/metrics`
- Metrics POST: `http://localhost:8000/api/metrics`
- Metrics GET: `http://localhost:8000/api/metrics`

## Integration with MAHIA

To integrate with MAHIA training, use the `TelemetryDashboardLogger` class from `telemetry_integration.py`:

```python
from dashboard.telemetry_integration import TelemetryDashboardLogger

# Create logger
logger = TelemetryDashboardLogger("http://localhost:8000/api/metrics")

# Log metrics during training
logger.log_loss(step, train_loss, val_loss)
logger.log_learning_rate(step, learning_rate)
logger.log_entropy(step, entropy)
```