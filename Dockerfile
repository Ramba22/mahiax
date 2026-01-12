# MAHIA-X Dockerfile
# Base image with CUDA support
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    rsync \
    openssh-client \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -tipsy

# Create conda environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean -tipsy && \
    rm /tmp/environment.yml

# Activate conda environment
SHELL ["conda", "run", "-n", "mahia-x", "/bin/bash", "-c"]

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install additional pip packages
RUN pip install --no-cache-dir -e .

# Expose ports for dashboard and API
EXPOSE 8501 8000

# Set entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mahia-x", "python"]

# Default command
CMD ["--help"]