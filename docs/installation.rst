Installation Guide
=================

MAHIA-X can be installed in several ways depending on your environment and requirements.

Prerequisites
-------------

Before installing MAHIA-X, ensure you have the following:

- Python 3.9 or higher
- CUDA 11.8 or higher (for GPU support)
- At least 8GB of RAM (16GB recommended)
- NVIDIA GPU with at least 8GB VRAM (for GPU training)

Conda Installation (Recommended)
--------------------------------

The easiest way to install MAHIA-X is using Conda:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-username/mahia-x.git
   cd mahia-x

   # Create conda environment
   conda env create -f environment.yml

   # Activate environment
   conda activate mahia-x

   # Install the package
   pip install -e .

Docker Installation
-------------------

MAHIA-X can also be run using Docker:

.. code-block:: bash

   # Build the Docker image
   docker build -t mahia-x .

   # Run the container
   docker run --gpus all -p 8501:8501 -p 8000:8000 mahia-x

Manual Installation
-------------------

If you prefer to install dependencies manually:

.. code-block:: bash

   # Create a virtual environment (optional but recommended)
   python -m venv mahia-x-env
   source mahia-x-env/bin/activate  # On Windows: mahia-x-env\Scripts\activate

   # Install core dependencies
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install numpy scipy pandas scikit-learn matplotlib seaborn
   pip install transformers datasets accelerate bitsandbytes torchao
   pip install onnx onnxruntime onnxruntime-gpu
   pip install streamlit fastapi uvicorn websocket-client
   pip install pynvml psutil

   # Install MAHIA-X
   pip install -e .

Verification
------------

To verify that MAHIA-X is installed correctly:

.. code-block:: bash

   python -c "import mahia_x; print('MAHIA-X imported successfully')"

You can also run the unit tests:

.. code-block:: bash

   python -m pytest tests/

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **CUDA Out of Memory**: Reduce batch size or use gradient checkpointing
2. **Import Errors**: Ensure all dependencies are installed correctly
3. **Permission Errors**: Run with appropriate permissions or use virtual environments

If you encounter any issues, please check the `GitHub Issues <https://github.com/your-username/mahia-x/issues>`_ page or create a new issue.