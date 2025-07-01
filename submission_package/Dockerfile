# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app
ENV OMP_NUM_THREADS=4

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY preprocess.py .
COPY train.py .
COPY inference.py .
COPY evaluate.py .
COPY README.md .

# Create necessary directories
RUN mkdir -p /app/data /app/preprocessed_data /app/checkpoints /app/results /app/output

# Set permissions
RUN chmod +x *.py

# Create a non-root user for security
RUN useradd -m -u 1000 eeguser && chown -R eeguser:eeguser /app
USER eeguser

# Set the default command
CMD ["python3", "--help"]

# Labels for metadata
LABEL maintainer="EEG Classification Team"
LABEL version="1.0"
LABEL description="Complete EEG classification pipeline for MI and SSVEP tasks"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python3 -c "import torch, sklearn, numpy, pandas; print('Dependencies OK')" || exit 1

# Example usage commands (as comments)
# Build: docker build -t eeg-classification .
# Run preprocessing: docker run -v /host/data:/app/data -v /host/output:/app/preprocessed_data eeg-classification python3 preprocess.py --data_path /app/data --output_path /app/preprocessed_data
# Run training: docker run --gpus all -v /host/preprocessed:/app/preprocessed_data -v /host/checkpoints:/app/checkpoints eeg-classification python3 train.py --data_path /app/preprocessed_data --save_path /app/checkpoints
# Run inference: docker run --gpus all -v /host/preprocessed:/app/preprocessed_data -v /host/checkpoints:/app/checkpoints -v /host/output:/app/output eeg-classification python3 inference.py --data_path /app/preprocessed_data --model_path /app/checkpoints --output_path /app/output/submission.csv
