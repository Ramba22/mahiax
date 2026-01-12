"""
MAHIA OptiCore - Memory and Performance Optimization Layer
Replacement for Triton with telemetry-guided, adaptive optimization.
"""

# Core OptiCore modules
from .core_manager import CoreManager, get_core_manager
from .memory_allocator import MemoryAllocator, get_memory_allocator
from .pooling_engine import PoolingEngine, get_pooling_engine
from .activation_checkpoint import ActivationCheckpointController, get_activation_checkpoint
from .precision_tuner import PrecisionTuner, get_precision_tuner
from .telemetry_layer import TelemetryLayer, get_telemetry_layer
from .energy_controller import EnergyController, get_energy_controller
from .diagnostics import Diagnostics, get_diagnostics
from .opticore import (OptiCore, get_opticore, initialize_opticore, shutdown_opticore,
                      opticore_memory, opticore_pooling, opticore_checkpoint,
                      opticore_precision, opticore_telemetry, opticore_energy,
                      opticore_diagnostics)

# Public API
__all__ = [
    "CoreManager",
    "MemoryAllocator", 
    "PoolingEngine",
    "ActivationCheckpointController",
    "PrecisionTuner",
    "TelemetryLayer",
    "EnergyController",
    "Diagnostics",
    "get_core_manager",
    "get_memory_allocator",
    "get_pooling_engine", 
    "get_activation_checkpoint",
    "get_precision_tuner",
    "get_telemetry_layer",
    "get_energy_controller",
    "get_diagnostics"
]

# Package version
__version__ = "1.0.0"
__author__ = "MAHIA Research Team"

print("🧩 MAHIA OptiCore initialized - Ready to replace Triton")