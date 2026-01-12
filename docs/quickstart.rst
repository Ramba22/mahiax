Quick Start Guide
================

This guide will help you get started with MAHIA-X quickly.

Basic Training
--------------

To start training a model with MAHIA-X:

.. code-block:: python

   from mahia_x import MAHIAExpertTrainer

   # Initialize trainer
   trainer = MAHIAExpertTrainer(
       model_config="config/model.yaml",
       data_config="config/data.yaml"
   )

   # Start training
   trainer.train()

Configuration Files
-------------------

MAHIA-X uses YAML configuration files for model and data configuration.

Example model configuration (config/model.yaml):

.. code-block:: yaml

   model_name: "MAHIA-X"
   version: "1.0"
   architecture:
     text_encoder:
       type: "Transformer-based"
       vocab_size: 30522
       embed_dim: 768
       max_seq_len: 512
       num_layers: 12
     moe_layer:
       type: "SparseMoETopK"
       num_experts: 8
       top_k: 2
       capacity_factor: 1.25
     output_layer:
       type: "MLP"
       hidden_dim: 768
       output_dim: 2

   training:
     mixed_precision: true
     gradient_checkpointing: true
     quantization:
       type: "FP8"
       enabled: true
     lora:
       enabled: true
       rank: 8
       alpha: 1.0

Example data configuration (config/data.yaml):

.. code-block:: yaml

   dataset:
     type: "huggingface"
     name: "glue"
     subset: "cola"
     split: "train"
   
   preprocessing:
     max_length: 512
     batch_size: 32
     shuffle: true
   
   augmentation:
     enabled: true
     techniques:
       - "token_masking"
       - "noise_injection"

Running Benchmarks
------------------

MAHIA-X includes a comprehensive benchmarking suite:

.. code-block:: bash

   # Run GLUE benchmarks
   mahia-eval --benchmark glue --model-path ./checkpoints/latest

   # Run custom benchmarks
   mahia-eval --benchmark custom --config benchmarks/custom.yaml

Using the Dashboard
-------------------

MAHIA-X includes a web-based dashboard for monitoring training:

.. code-block:: bash

   # Start the dashboard
   mahia-dashboard --port 8501

Then open your browser to http://localhost:8501 to access the dashboard.

Exporting Models
----------------

Export models to various formats for deployment:

.. code-block:: bash

   # Export to ONNX
   mahia-export --format onnx --model-path ./checkpoints/latest --output-path ./exported/

   # Export to TorchScript
   mahia-export --format torchscript --model-path ./checkpoints/latest --output-path ./exported/