"""
Streamlit Frontend for MAHIA Dashboard V3
Real-time visualization of training metrics
"""

import streamlit as st
import asyncio
import websockets
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
from typing import Dict, Any, List

# Streamlit page configuration
st.set_page_config(
    page_title="MAHIA Dashboard V3",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'metrics_data' not in st.session_state:
    st.session_state.metrics_data = {}
    
if 'connected' not in st.session_state:
    st.session_state.connected = False
    
if 'websocket' not in st.session_state:
    st.session_state.websocket = None

def initialize_dashboard():
    """Initialize the dashboard layout"""
    st.title("🚀 MAHIA Dashboard V3")
    st.markdown("Real-time monitoring of MAHIA training metrics")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Connection settings
        st.subheader("Connection")
        server_url = st.text_input("WebSocket Server URL", "ws://localhost:8000/ws/metrics")
        
        # Connection button
        if not st.session_state.connected:
            if st.button("🔌 Connect to Server"):
                connect_to_server(server_url)
        else:
            if st.button("🔌 Disconnect"):
                disconnect_from_server()
                
        # Display connection status
        status = "🟢 Connected" if st.session_state.connected else "🔴 Disconnected"
        st.markdown(f"**Status:** {status}")
        
        # Refresh rate
        refresh_rate = st.slider("Refresh Rate (ms)", 100, 2000, 500)
        
        # Metrics selection
        st.subheader("📊 Metrics to Display")
        show_loss = st.checkbox("Loss Curves", True)
        show_lr = st.checkbox("Learning Rate", True)
        show_entropy = st.checkbox("Entropy Metrics", True)
        show_gpu = st.checkbox("GPU Utilization", True)
        show_controller = st.checkbox("Controller Actions", True)
        
    return {
        "refresh_rate": refresh_rate,
        "show_loss": show_loss,
        "show_lr": show_lr,
        "show_entropy": show_entropy,
        "show_gpu": show_gpu,
        "show_controller": show_controller
    }

async def connect_websocket(url: str):
    """Connect to WebSocket server"""
    try:
        websocket = await websockets.connect(url)
        st.session_state.websocket = websocket
        st.session_state.connected = True
        st.success("Connected to WebSocket server")
        return websocket
    except Exception as e:
        st.error(f"Failed to connect: {e}")
        st.session_state.connected = False
        return None

def connect_to_server(url: str):
    """Connect to the WebSocket server"""
    try:
        # This is a simplified approach - in practice, you'd need to handle async properly
        st.session_state.connected = True
        st.success("Connected to WebSocket server")
    except Exception as e:
        st.error(f"Failed to connect: {e}")

def disconnect_from_server():
    """Disconnect from the WebSocket server"""
    st.session_state.connected = False
    st.session_state.websocket = None
    st.success("Disconnected from WebSocket server")

def update_metrics_display(settings: Dict[str, Any]):
    """Update the metrics display"""
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    # Display loss curves
    if settings["show_loss"] and "loss" in st.session_state.metrics_data:
        with col1:
            st.subheader("📉 Loss Curves")
            loss_data = st.session_state.metrics_data["loss"]
            if loss_data:
                df = pd.DataFrame(loss_data)
                fig = px.line(df, x="step", y=["train_loss", "val_loss"], 
                             title="Training and Validation Loss")
                st.plotly_chart(fig, use_container_width=True)
    
    # Display learning rate
    if settings["show_lr"] and "learning_rate" in st.session_state.metrics_data:
        with col2:
            st.subheader("🧠 Learning Rate")
            lr_data = st.session_state.metrics_data["learning_rate"]
            if lr_data:
                df = pd.DataFrame(lr_data)
                fig = px.line(df, x="step", y="lr", title="Learning Rate Schedule")
                st.plotly_chart(fig, use_container_width=True)
    
    # Display entropy metrics
    if settings["show_entropy"] and "entropy" in st.session_state.metrics_data:
        with col1:
            st.subheader("🌀 Entropy Metrics")
            entropy_data = st.session_state.metrics_data["entropy"]
            if entropy_data:
                df = pd.DataFrame(entropy_data)
                fig = px.line(df, x="step", y=["gradient_entropy", "confidence"], 
                             title="Gradient Entropy and Confidence")
                st.plotly_chart(fig, use_container_width=True)
    
    # Display GPU utilization
    if settings["show_gpu"] and "gpu" in st.session_state.metrics_data:
        with col2:
            st.subheader("🖥️ GPU Utilization")
            gpu_data = st.session_state.metrics_data["gpu"]
            if gpu_data:
                df = pd.DataFrame(gpu_data)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["step"], y=df["gpu_util"], 
                                       mode='lines', name='GPU Utilization (%)'))
                fig.add_trace(go.Scatter(x=df["step"], y=df["memory_used"], 
                                       mode='lines', name='Memory Used (MB)'))
                fig.update_layout(title="GPU Utilization and Memory Usage")
                st.plotly_chart(fig, use_container_width=True)
    
    # Display controller actions
    if settings["show_controller"] and "controller" in st.session_state.metrics_data:
        with col1:
            st.subheader("🎮 Controller Actions")
            controller_data = st.session_state.metrics_data["controller"]
            if controller_data:
                df = pd.DataFrame(controller_data)
                action_counts = df["action"].value_counts()
                fig = px.bar(x=action_counts.index, y=action_counts.values,
                           labels={'x': 'Action', 'y': 'Count'},
                           title="Controller Action Distribution")
                st.plotly_chart(fig, use_container_width=True)

def simulate_metrics_data():
    """Simulate metrics data for demonstration"""
    # This is just for demonstration - in real implementation, data would come from WebSocket
    step = len(st.session_state.metrics_data.get("loss", []))
    
    # Simulate loss data
    if "loss" not in st.session_state.metrics_data:
        st.session_state.metrics_data["loss"] = []
    
    train_loss = max(0.1, 1.0 - step * 0.01 + 0.1 * (1 - step/100))
    val_loss = max(0.15, 1.1 - step * 0.008 + 0.15 * (1 - step/100))
    
    st.session_state.metrics_data["loss"].append({
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "timestamp": time.time()
    })
    
    # Simulate learning rate data
    if "learning_rate" not in st.session_state.metrics_data:
        st.session_state.metrics_data["learning_rate"] = []
        
    lr = 0.001 * (0.95 ** (step // 10))
    
    st.session_state.metrics_data["learning_rate"].append({
        "step": step,
        "lr": lr,
        "timestamp": time.time()
    })
    
    # Simulate entropy data
    if "entropy" not in st.session_state.metrics_data:
        st.session_state.metrics_data["entropy"] = []
        
    entropy = max(0.1, 1.0 - step * 0.005)
    confidence = min(0.99, 0.5 + step * 0.003)
    
    st.session_state.metrics_data["entropy"].append({
        "step": step,
        "gradient_entropy": entropy,
        "confidence": confidence,
        "timestamp": time.time()
    })
    
    # Simulate GPU data
    if "gpu" not in st.session_state.metrics_data:
        st.session_state.metrics_data["gpu"] = []
        
    gpu_util = min(100, 30 + step * 0.5)
    memory_used = min(24000, 1000 + step * 20)
    
    st.session_state.metrics_data["gpu"].append({
        "step": step,
        "gpu_util": gpu_util,
        "memory_used": memory_used,
        "temperature": 65 + step * 0.1,
        "timestamp": time.time()
    })
    
    # Simulate controller data
    if "controller" not in st.session_state.metrics_data:
        st.session_state.metrics_data["controller"] = []
        
    actions = ["wait", "extend", "stop"]
    action = actions[step % 3]
    
    st.session_state.metrics_data["controller"].append({
        "step": step,
        "action": action,
        "confidence": confidence,
        "timestamp": time.time()
    })

def main():
    """Main dashboard function"""
    settings = initialize_dashboard()
    
    # Simulate metrics data for demonstration
    simulate_metrics_data()
    
    # Update display
    update_metrics_display(settings)
    
    # Auto-refresh
    time.sleep(settings["refresh_rate"] / 1000)
    st.experimental_rerun()

if __name__ == "__main__":
    main()