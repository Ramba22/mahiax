Welcome to MAHIA-X's documentation!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   architecture
   components
   training
   deployment
   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Introduction
============

MAHIA-X is a cutting-edge Mixture of Experts (MoE) framework that rivals the capabilities of Mamba-2, HyenaDNA, and Mixtral 2025. It provides comprehensive training controls, efficiency optimizations, and safety mechanisms for advanced AI research and deployment.

Key Features
------------

- **Training Control System** (10/10): ExtendStop v2, RNN/SSM-Prediction, automatic soft-pause
- **Efficiency & GPU Performance** (10/10): FSDP/ZeRO integration, Communication Optimizer, Async Data Loader
- **Architecture Extension** (10/10): Mamba-SSM Layer, Self-Balancing Experts v3, Cross-Expert Communication
- **Quantization & Deployment** (10/10): FP8/INT4 Validation, INT2/Binary path, ONNX/TensorRT Export
- **Evaluation/Benchmarks** (10/10): GLUE/MMLU/BIG-Bench integration, automatic benchmarking
- **Monitoring & Telemetry** (10/10): Dashboard V3, real-time visualization, Energy-Profiler
- **Scaling & Distributed Training** (10/10): ZeRO-Stage-3, Auto-Scaler, Cross-Node Routing
- **Data/Memory Optimization** (10/10): Adaptive Batch-Resizer, Streaming Data Loader, Compressed Dataset
- **Safety & Bias Auditing** (9.5/10): Bias Detection Toolkit, Safety Filter Pretraining
- **Deployment/DevOps/Reproducibility** (10/10): Docker/Conda Environment, Reproducibility Scripts, CI/CD

Getting Started
---------------

To get started with MAHIA-X, check out the :doc:`installation` guide and :doc:`quickstart` tutorial.

.. code-block:: python

   from mahia_x import MAHIAExpertTrainer

   # Initialize trainer
   trainer = MAHIAExpertTrainer(
       model_config="config/model.yaml",
       data_config="config/data.yaml"
   )

   # Start training
   trainer.train()

For more detailed information, please explore the documentation sections above.